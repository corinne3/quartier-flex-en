"""
rules.py: the strategies WITHOUT artificial intelligence.

- NoBattery          : "panels only" reference (the simulator has no battery).
- SelfConsumptionRule: the battery stores the solar surplus and gives it back as soon
                       as electricity is lacking. No forecast. This is what
                       most off-the-shelf installations do.
- TariffRule         : in addition, charges from the grid during off-peak hours at night
                       if the forecast sun ("same as yesterday") will not recharge the
                       battery enough to cover the next day's peak hours, and
                       keeps the energy for peak hours.
                       Naive forecast, no AI.
"""

from __future__ import annotations

import numpy as np

from .base import Controller, Decision, State


class NoBattery(Controller):
    name = "0_no_battery"
    label = "Panels without battery"
    family = "reference"
    description = "Solar panels only: the surplus is sold, the shortfall is bought."

    def decide(self, state: State) -> Decision:
        return Decision(0.0, 0.0, "no battery")


class SelfConsumptionRule(Controller):
    name = "1_rule_autoconso"
    label = "Self-consumption rule"
    family = "rule"
    description = "Stores the solar surplus, gives it back as soon as power is lacking. No forecast."

    def decide(self, state: State) -> Decision:
        return Decision(reason="follow the surplus")


class TariffRule(Controller):
    name = "2_tariff_rule"
    label = "Tariff rule"
    family = "rule"
    description = ("Self-consumption + overnight off-peak recharge if the forecast solar "
                   "(\"same as yesterday\") will not cover the next day's peak hours.")

    def setup(self, ctx, meter) -> None:
        super().setup(ctx, meter)
        self.hc_hours = set(ctx.scenario.tariff.hc_hours)

    def decide(self, state: State) -> Decision:
        h = state.t.hour
        if h in self.hc_hours:
            hz = state.horizon.iloc[:24]
            day = hz[~hz.index.hour.isin(self.hc_hours)]
            net = day["load_fc"] - day["pv_fc"]
            deficit = float(np.clip(net, 0, None).sum())       # what will be lacking during peak hours
            solar_refill = float(np.clip(-net, 0, None).sum()) * 0.9  # what the sun will recharge
            target = float(np.clip(deficit - solar_refill, 0, state.usable_kwh))
            missing = max(0.0, target - (state.energy_kwh - 0.1 * state.capacity_kwh))
            n_hc_left = sum(1 for t in hz.index[:8] if t.hour in self.hc_hours)
            if target <= 0:
                return Decision(reason="off-peak: the sun will suffice tomorrow, self-consumption")
            p = min(state.max_charge_kw, missing / max(n_hc_left, 1))
            return Decision(grid_charge_kw=p, max_discharge_kw=0.0, reason=f"off-peak: target {target:.0f} kWh")
        return Decision(reason="peak: self-consumption")
