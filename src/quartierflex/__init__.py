"""
quartierflex: demand response for a district (connected appliances, second-life electric-car batteries, solar),
and the challenge question: does the AI that optimizes consume more than it saves?

Modules (each testable on its own):
    1 weather          observed and forecast weather, sun position
    2 solar            panel output, meter, AI forecast
    3 batteries        second-life batteries: model, aging, BMS (SOC/SOH estimation)
    4 usage            home consumption by occupant type, habits, AI forecast
    5 appliances       heating, water heater, EV: sheddable groups, learned thermal model
    6 grid             tariffs (Tempo), resale, CO2 and national consumption (RTE)
    7 demand_response  RTE requests, declared flexibility (batteries included)
    8 control          batteries serving self-consumption (rules, optimizer, LLM agents)
    9 aggregator       allocation of the request across groups and batteries, net AI balance
   10 measure          energy cost of the AI
   11 assessment       hourly simulation (self-consumption), sizing
"""

__version__ = "0.1.0"
