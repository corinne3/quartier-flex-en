"""
config.py: ALL the project's assumptions, in one place.

Each block maps to a module:
    Site                 -> weather, solar
    PVConfig             -> solar
    BatteryConfig        -> batteries
    BuildingConfig       -> usage
    AppliancesConfig     -> appliances (heating, hot water, electric car)
    TariffConfig         -> grid
    DemandResponseConfig -> demand_response (RTE requests) and aggregator
    ComputeParams        -> measure
    Scenario             -> assessment (puts everything together)

Rule: every value is commented. "TO BE CHECKED" = order of magnitude to be
confirmed by the team (the researchers!) before presenting the results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"      # downloaded data (not versioned)
RESULTS_DIR = ROOT / "results"


# =============================================================================
# Location
# =============================================================================
@dataclass
class Site:
    name: str = "Valence"
    lat: float = 44.93
    lon: float = 4.89


# =============================================================================
# Solar panels
# =============================================================================
@dataclass
class PVConfig:
    """
    kwc = kilowatt-peak (kWp): panel power under "standard" sunlight
    (1,000 W/m², 25 °C). It is THE sizing variable.
    Order of magnitude: 1 panel ≈ 0.4 kWp ≈ 2 m².
    """

    kwc: float = 120.0         # district of 60 homes: ~2 kWp per home (rooftops)
    tilt_deg: float = 30.0     # tilt (0 = flat, 90 = vertical)
    azimuth_deg: float = 180.0 # orientation: 180 = due south
    gamma_per_c: float = -0.004  # efficiency loss per °C above 25 °C (silicon)
    noct_c: float = 45.0       # nominal operating cell temperature
    performance_ratio: float = 0.86  # system losses: inverter, cables, dust... (0.8-0.9)
    albedo: float = 0.2        # ground reflectance
    price_eur_per_kwc: float = 1500.0  # installed cost (panels + inverter + installation) — TO BE CHECKED


# =============================================================================
# Second-life batteries
# =============================================================================
@dataclass
class BatteryConfig:
    """
    REUSED electric-car batteries.
    A car battery is removed from the vehicle at around 70-80% of its
    original capacity (state of health, SOH). It can still serve in
    stationary use down to ~60%. TO BE CHECKED against the team's sources.

    Default example: 4 packs of ~40 kWh when new (Renault Zoe / Nissan
    Leaf size), recovered at between 71% and 78% state of health.
    """

    capacities_kwh: list[float] = field(default_factory=lambda: [40.0] * 4)   # capacity of each pack when NEW
    soh_init: list[float] = field(default_factory=lambda: [0.78, 0.71, 0.76, 0.73])  # state of health at installation
    p_max_kw: float = 7.0          # max power per pack (limited by the inverter)
    eta_charge: float = 0.95       # charge efficiency
    eta_discharge: float = 0.95    # discharge efficiency (round trip ≈ 90%)
    soc_min: float = 0.10          # never go below 10% (preserves the battery)
    soc_max: float = 0.90          # nor above 90%
    aging_per_efc: float = 0.00012 # SOH loss per equivalent full cycle (0.012%) — TO BE CHECKED
    aging_calendar_per_year: float = 0.015  # SOH loss per year, even when idle — TO BE CHECKED
    price_eur_per_kwh: float = 90.0  # second-life purchase price (€/kWh of capacity) — TO BE CHECKED


# =============================================================================
# Homes
# =============================================================================
@dataclass
class BuildingConfig:
    n_homes: int = 60      # the DISTRICT (several buildings): the scale of demand response
    # Mix of occupant types (see usage/profiles.py). Must sum to 1.
    mix: dict = field(default_factory=lambda: {
        "family": 0.35, "working_couple": 0.25, "retirees": 0.2, "remote_worker": 0.15, "student": 0.05,
    })


# =============================================================================
# Connected appliances in the homes
# =============================================================================
@dataclass
class AppliancesConfig:
    """
    Who has what? Each home has an occupant TYPE (usage module) and
    EQUIPMENT (drawn at random with these shares). Only homes fitted with a
    CONNECTED box (smart controller) can be shed.
    Shares = orders of magnitude TO BE CHECKED (researchers: ADEME, CEREN, Enedis).
    """

    share_heating_electric: float = 0.55   # homes with electric heating (heaters, heat pumps)
    share_water_heater_electric: float = 0.65      # electric water heater
    share_ev: float = 0.25               # electric car (EV) charged at home
    share_connected: float = 0.80        # homes with a controllable box (otherwise not sheddable)
    share_water_heater_off_peak: float = 0.70  # water heaters tied to off-peak hours (otherwise heat at any time)
    heat_loss_w_per_k_m2: float = 1.2 # insulation: losses (W per °C of difference per m²) — 0.6 = well insulated, 2.5 = leaky
    inertia_kwh_per_k_m2: float = 0.035 # thermal inertia (kWh to heat 1 m² by 1 °C, walls included)
    p_heating_kw_per_m2: float = 0.08 # installed heater power (80 W/m²)
    tolerance_c: float = 1.0            # comfort: dropping to setpoint − 1 °C is accepted when occupants are home
    tolerance_vulnerable_c: float = 0.5 # retirees: reduced tolerance (more vulnerable to cold)
    water_heater_kw: float = 2.2                 # power of one water heater
    water_heater_kwh_max: float = 10.0           # energy stored in a full water heater (~200 L)
    ev_kw: float = 7.0                  # 7 kW charging point
    ev_kwh_per_day: float = 9.0        # ~50 km per day


# =============================================================================
# Grid: tariffs
# =============================================================================
@dataclass
class TariffConfig:
    """
    Price per kWh including taxes (excluding subscription). 2025 ORDERS OF
    MAGNITUDE — TO BE CHECKED on the supplier's website before presenting euros.
    """

    option: str = "TEMPO"                # "BASE", "HPHC" or "TEMPO"
    base: float = 0.20
    hp: float = 0.21
    hc: float = 0.17
    tempo: dict = field(default_factory=lambda: {  # color: (off-peak, peak)
        "BLUE": (0.13, 0.16), "WHITE": (0.15, 0.18), "RED": (0.15, 0.66),
    })
    hc_hours: tuple = (22, 23, 0, 1, 2, 3, 4, 5)  # off-peak hours (8 h/day)
    sell_surplus: float = 0.04           # feed-in price for exported surplus — TO BE CHECKED


# =============================================================================
# Demand response: RTE requests
# =============================================================================
@dataclass
class DemandResponseConfig:
    """
    How it works in France (to explain to the jury):
    - RTE (the French transmission system operator) does NOT control homes. It
      activates an AGGREGATOR ("demand-response operator") that has committed
      to a volume: "shed X kW from 6 pm to 8 pm".
    - The aggregator (our AI) allocates this volume among its homes, cutting
      appliances for a few minutes, in turn.
    - The aggregator continuously declares its available FLEXIBILITY (batteries +
      appliances): RTE sizes its requests based on it.
    Values = orders of magnitude TO BE CHECKED.
    """

    slots: tuple = ((7, 9), (18, 20))    # national winter peaks (start, end hours) — see EcoWatt
    share_of_flex: float = 0.6            # RTE requests 60% of the flexibility declared the day before
    duration_max_cut_min: int = 30         # an appliance is never cut for more than 30 min in a row
    rest_min: int = 30                     # ... and then stays on for at least 30 min
    payment_eur_mwh: float = 200.0     # payment for shed energy — TO BE CHECKED
    penalty_eur_mwh: float = 300.0         # penalty for promised energy not shed — TO BE CHECKED
    step_min: int = 15                       # simulation time step (minutes)


# =============================================================================
# Measuring the cost of the AI
# =============================================================================
@dataclass
class ComputeParams:
    cpu_tdp_w: float = 28.0    # CPU TDP (i5-1155G7: 28 W)
    idle_power_w: float = 10.0
    pue: float = 1.0


# =============================================================================
# Full scenario
# =============================================================================
@dataclass
class Scenario:
    site: Site = field(default_factory=Site)
    pv: PVConfig = field(default_factory=PVConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    building: BuildingConfig = field(default_factory=BuildingConfig)
    appliances: AppliancesConfig = field(default_factory=AppliancesConfig)
    tariff: TariffConfig = field(default_factory=TariffConfig)
    demand_response: DemandResponseConfig = field(default_factory=DemandResponseConfig)
    compute: ComputeParams = field(default_factory=ComputeParams)
    start: str = "2024-01-15"   # start of the simulated period (winter: when RTE requests demand response)
    days: int = 7
    history_days: int = 60      # history used to train the models (never the future)
    seed: int = 42
    offline: bool = False       # True = no downloads (synthetic data)
