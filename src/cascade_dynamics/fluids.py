from __future__ import annotations

from CoolProp.CoolProp import PropsSI


def h_tp(temperature_k: float, pressure_pa: float, fluid: str) -> float:
    return float(PropsSI("H", "T", temperature_k, "P", pressure_pa, fluid))


def s_tp(temperature_k: float, pressure_pa: float, fluid: str) -> float:
    return float(PropsSI("S", "T", temperature_k, "P", pressure_pa, fluid))


def h_ps(pressure_pa: float, entropy_j_kg_k: float, fluid: str) -> float:
    return float(PropsSI("H", "P", pressure_pa, "S", entropy_j_kg_k, fluid))


def p_sat(temperature_k: float, fluid: str) -> float:
    return float(PropsSI("P", "T", temperature_k, "Q", 0.0, fluid))


def t_sat(pressure_pa: float, fluid: str) -> float:
    return float(PropsSI("T", "P", pressure_pa, "Q", 0.0, fluid))


def h_sat_liq(temperature_k: float, fluid: str) -> float:
    return float(PropsSI("H", "T", temperature_k, "Q", 0.0, fluid))


def h_refrigerant_liquid(temperature_k: float, pressure_pa: float, fluid: str, subcooling_k: float) -> float:
    if subcooling_k <= 1.0e-9:
        return h_sat_liq(temperature_k, fluid)
    return h_tp(temperature_k, pressure_pa, fluid)
