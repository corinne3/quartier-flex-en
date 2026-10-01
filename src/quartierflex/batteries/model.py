"""
model.py: electric-car batteries in their SECOND LIFE.

Vocabulary (worth knowing for the pitch)
----------------------------------------
- Nominal capacity: what the battery stored when NEW (e.g. 40 kWh).
- SOH (State of Health): current capacity / capacity when new.
      A battery retired from a car typically has an SOH of 70-80%.
- SOC (State of Charge): fill level, from 0 to 100% of the CURRENT
      capacity (like a phone's battery gauge).
- Equivalent full cycle (EFC): amount of energy that has flowed through the
      battery, expressed as "one full charge + one full discharge".

What the model represents
-------------------------
1. Usable capacity = nominal capacity × SOH, and it is only used
   between SOC min and SOC max (10%-90%) to spare the battery.
2. Efficiency: charging 1 kWh stores only 0.95; delivering 1 kWh
   drains 1/0.95. Round trip ≈ 90%.
3. Power: limited by the inverter, and REDUCED if the battery is worn
   (an aged battery heats up more; we spare it):
       effective P_max = P_max × min(1, SOH / 0.8)
4. Aging (SOH decreases):
   - through use  : ΔSOH = coefficient × equivalent full cycles;
   - through time : ΔSOH = annual loss × duration, faster if the
     battery stays highly charged (factor 0.5 + SOC).
5. A BANK of several packs with different health: the requested power
   is split in proportion to what each pack can still deliver or
   absorb. The most worn packs work less: this is how the
   "battery condition is taken into account".

Sign convention: power > 0 = CHARGE, < 0 = DISCHARGE (on the building's
electrical grid side, in average kW over the hour = kWh).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import BatteryConfig


@dataclass
class BatteryPack:
    name: str
    cap_name_kwh: float
    soh: float
    cfg: BatteryConfig
    soc: float = 0.5                # fraction of the CURRENT capacity
    throughput_kwh: float = 0.0     # cumulative energy that has flowed through the pack
    history: list = field(default_factory=list)

    # ------------------------------------------------------------ properties
    @property
    def capacity_kwh(self) -> float:
        return self.cap_name_kwh * self.soh

    @property
    def energy_kwh(self) -> float:
        return self.soc * self.capacity_kwh

    @property
    def efc(self) -> float:
        """Equivalent full cycles since installation."""
        return self.throughput_kwh / (2 * self.cap_name_kwh)

    @property
    def p_max_kw(self) -> float:
        return self.cfg.p_max_kw * min(1.0, self.soh / 0.8)

    def max_charge_kw(self, dt_h: float = 1.0) -> float:
        room = max(0.0, (self.cfg.soc_max - self.soc) * self.capacity_kwh)
        return min(self.p_max_kw, room / (self.cfg.eta_charge * dt_h))

    def max_discharge_kw(self, dt_h: float = 1.0) -> float:
        avail = max(0.0, (self.soc - self.cfg.soc_min) * self.capacity_kwh)
        return min(self.p_max_kw, avail * self.cfg.eta_discharge / dt_h)

    # ------------------------------------------------------------ one hour
    def step(self, power_kw: float, dt_h: float = 1.0) -> float:
        """Applies a setpoint; returns the power ACTUALLY exchanged (bounded)."""
        if power_kw >= 0:
            p = min(power_kw, self.max_charge_kw(dt_h))
            stored = p * dt_h * self.cfg.eta_charge
        else:
            p = -min(-power_kw, self.max_discharge_kw(dt_h))
            stored = p * dt_h / self.cfg.eta_discharge
        cap = self.capacity_kwh
        self.soc = float(np.clip(self.soc + stored / cap, 0.0, 1.0))
        self.throughput_kwh += abs(stored)
        # Aging
        d_efc = abs(stored) / (2 * self.cap_name_kwh)
        d_cal = self.cfg.aging_calendar_per_year * dt_h / 8760 * (0.5 + self.soc)
        self.soh = max(0.0, self.soh - self.cfg.aging_per_efc * d_efc - d_cal)
        return p


class BatteryBank:
    """Several second-life packs, controlled as a single battery."""

    def __init__(self, cfg: BatteryConfig, soc0: float = 0.5):
        self.cfg = cfg
        self.packs = [
            BatteryPack(f"pack_{i + 1}", c, s, cfg, soc=soc0)
            for i, (c, s) in enumerate(zip(cfg.capacities_kwh, cfg.soh_init))
        ]
        self.soh0 = [p.soh for p in self.packs]

    # Aggregates used for control
    @property
    def capacity_kwh(self) -> float:
        return sum(p.capacity_kwh for p in self.packs)

    @property
    def energy_kwh(self) -> float:
        return sum(p.energy_kwh for p in self.packs)

    @property
    def soc(self) -> float:
        cap = self.capacity_kwh
        return self.energy_kwh / cap if cap > 0 else 0.0

    @property
    def usable_kwh(self) -> float:
        return sum((self.cfg.soc_max - self.cfg.soc_min) * p.capacity_kwh for p in self.packs)

    def max_charge_kw(self) -> float:
        return sum(p.max_charge_kw() for p in self.packs)

    def max_discharge_kw(self) -> float:
        return sum(p.max_discharge_kw() for p in self.packs)

    def step(self, power_kw: float, dt_h: float = 1.0) -> float:
        """Splits the setpoint between packs in proportion to their margin; returns the actual power."""
        if abs(power_kw) < 1e-9 or not self.packs:
            for p in self.packs:
                p.step(0.0, dt_h)
            return 0.0
        charge = power_kw > 0
        margins = np.array([p.max_charge_kw(dt_h) if charge else p.max_discharge_kw(dt_h) for p in self.packs])
        total = margins.sum()
        if total <= 1e-9:
            for p in self.packs:
                p.step(0.0, dt_h)
            return 0.0
        target = min(abs(power_kw), total)
        shares = margins / total * target * (1 if charge else -1)
        return float(sum(p.step(s, dt_h) for p, s in zip(self.packs, shares)))

    def soh_loss(self) -> list[float]:
        return [s0 - p.soh for s0, p in zip(self.soh0, self.packs)]
