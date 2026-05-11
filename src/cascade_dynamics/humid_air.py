from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


R_DRY_AIR = 287.055
R_WATER_VAPOR = 461.52
EPSILON = 0.621945
CP_DRY_AIR = 1004.6925
T_REF_K = 273.15
P_REF_PA = 101325.0
CP_WATER_VAPOR = 1860.0
CP_ICE = 2100.0
H_VAPOR_0C = 2_501_000.0
H_ICE_0C = -333_700.0
MIN_T_K = 150.0
MAX_T_K = 450.0


@dataclass(frozen=True)
class HumidAirState:
    temperature_k: float
    pressure_pa: float
    humidity_ratio_in: float
    vapor_humidity_ratio: float
    ice_humidity_ratio: float
    enthalpy_j_kg_da: float
    entropy_j_kg_da_k: float
    dry_air_density_kg_m3: float
    mixture_density_kg_m3: float
    saturation_humidity_ratio: float


def saturation_pressure_water_pa(temperature_k: float) -> float:
    """ASHRAE saturation pressure over ice below 0 C and water above 0 C."""
    t = max(float(temperature_k), 173.15)
    if t <= T_REF_K:
        ln_p_ws = (
            -5.6745359e3 / t
            + 6.3925247
            - 9.677843e-3 * t
            + 6.2215701e-7 * t**2
            + 2.0747825e-9 * t**3
            - 9.484024e-13 * t**4
            + 4.1635019 * math.log(t)
        )
    else:
        ln_p_ws = (
            -5.8002206e3 / t
            + 1.3914993
            - 4.8640239e-2 * t
            + 4.1764768e-5 * t**2
            - 1.4452093e-8 * t**3
            + 6.5459673 * math.log(t)
        )
    return math.exp(ln_p_ws)


def humidity_ratio_from_vapor_pressure(vapor_pressure_pa: float, pressure_pa: float) -> float:
    p_v = min(max(float(vapor_pressure_pa), 0.0), 0.999 * float(pressure_pa))
    return EPSILON * p_v / max(float(pressure_pa) - p_v, 1.0e-9)


def vapor_pressure_from_humidity_ratio(humidity_ratio: float, pressure_pa: float) -> float:
    x = max(float(humidity_ratio), 0.0)
    return float(pressure_pa) * x / (EPSILON + x)


def saturation_humidity_ratio(temperature_k: float, pressure_pa: float) -> float:
    return humidity_ratio_from_vapor_pressure(saturation_pressure_water_pa(temperature_k), pressure_pa)


def saturated_room_humidity_ratio(temperature_k: float, pressure_pa: float, relative_humidity: float = 1.0) -> float:
    p_ws = saturation_pressure_water_pa(temperature_k)
    p_v = min(max(float(relative_humidity), 0.0), 1.0) * p_ws
    return humidity_ratio_from_vapor_pressure(p_v, pressure_pa)


def _water_vapor_enthalpy(temperature_k: float) -> float:
    return H_VAPOR_0C + CP_WATER_VAPOR * (temperature_k - T_REF_K)


def _ice_enthalpy(temperature_k: float) -> float:
    return H_ICE_0C + CP_ICE * (temperature_k - T_REF_K)


def _water_vapor_entropy(temperature_k: float, vapor_pressure_pa: float) -> float:
    p_v = max(float(vapor_pressure_pa), 1.0e-9)
    return CP_WATER_VAPOR * math.log(temperature_k / T_REF_K) - R_WATER_VAPOR * math.log(p_v / P_REF_PA)


def _ice_entropy(temperature_k: float) -> float:
    return (H_ICE_0C / T_REF_K) + CP_ICE * math.log(temperature_k / T_REF_K)


def _dry_air_enthalpy(temperature_k: float) -> float:
    return CP_DRY_AIR * (temperature_k - T_REF_K)


def _dry_air_entropy(temperature_k: float, dry_air_pressure_pa: float) -> float:
    p_da = max(float(dry_air_pressure_pa), 1.0e-9)
    return CP_DRY_AIR * math.log(temperature_k / T_REF_K) - R_DRY_AIR * math.log(p_da / P_REF_PA)


def humid_air_state(temperature_k: float, pressure_pa: float, humidity_ratio: float) -> HumidAirState:
    t = float(temperature_k)
    p = float(pressure_pa)
    x_in = max(float(humidity_ratio), 0.0)
    x_sat = saturation_humidity_ratio(t, p)
    x_v = min(x_in, x_sat)
    x_ice = max(x_in - x_v, 0.0)
    p_v = vapor_pressure_from_humidity_ratio(x_v, p)
    p_da = max(p - p_v, 1.0e-9)

    h_air = _dry_air_enthalpy(t)
    s_air = _dry_air_entropy(t, p_da)
    h_total = h_air + x_v * _water_vapor_enthalpy(t) + x_ice * _ice_enthalpy(t)
    s_total = s_air + x_v * _water_vapor_entropy(t, p_v) + x_ice * _ice_entropy(t)
    rho_da = p_da / (R_DRY_AIR * t)
    rho_v = p_v / (R_WATER_VAPOR * t)

    return HumidAirState(
        temperature_k=t,
        pressure_pa=p,
        humidity_ratio_in=x_in,
        vapor_humidity_ratio=x_v,
        ice_humidity_ratio=x_ice,
        enthalpy_j_kg_da=h_total,
        entropy_j_kg_da_k=s_total,
        dry_air_density_kg_m3=rho_da,
        mixture_density_kg_m3=rho_da + rho_v,
        saturation_humidity_ratio=x_sat,
    )


def solve_temperature_for_property(
    pressure_pa: float,
    humidity_ratio: float,
    target: float,
    property_fn: Callable[[HumidAirState], float],
    lower_k: float = MIN_T_K,
    upper_k: float = MAX_T_K,
) -> HumidAirState:
    def residual(t_k: float) -> float:
        return property_fn(humid_air_state(t_k, pressure_pa, humidity_ratio)) - target

    lo = lower_k
    hi = upper_k
    f_lo = residual(lo)
    f_hi = residual(hi)
    expansion = 0
    while f_lo * f_hi > 0.0 and expansion < 12:
        if abs(f_lo) < abs(f_hi):
            hi = min(hi + 50.0, 800.0)
            f_hi = residual(hi)
        else:
            lo = max(lo - 25.0, 80.0)
            f_lo = residual(lo)
        expansion += 1
    if f_lo * f_hi > 0.0:
        raise ValueError("Could not bracket humid-air property temperature.")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) <= 1.0e-5 or (hi - lo) <= 1.0e-7:
            return humid_air_state(mid, pressure_pa, humidity_ratio)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return humid_air_state(0.5 * (lo + hi), pressure_pa, humidity_ratio)


def state_at_entropy(pressure_pa: float, humidity_ratio: float, entropy_j_kg_da_k: float) -> HumidAirState:
    return solve_temperature_for_property(
        pressure_pa,
        humidity_ratio,
        entropy_j_kg_da_k,
        lambda state: state.entropy_j_kg_da_k,
    )


def state_at_enthalpy(pressure_pa: float, humidity_ratio: float, enthalpy_j_kg_da: float) -> HumidAirState:
    return solve_temperature_for_property(
        pressure_pa,
        humidity_ratio,
        enthalpy_j_kg_da,
        lambda state: state.enthalpy_j_kg_da,
    )
