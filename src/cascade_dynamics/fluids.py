from __future__ import annotations

from functools import lru_cache

from CoolProp.CoolProp import ALTERNATIVE_REFPROP_PATH, PropsSI, set_config_string


DEFAULT_REFPROP_PATH = r"C:\Program Files (x86)\REFPROP"
_BACKEND = "coolprop"
_REFPROP_PATH = DEFAULT_REFPROP_PATH


def configure_property_backend(backend: str = "coolprop", refprop_path: str | None = None) -> None:
    global _BACKEND, _REFPROP_PATH

    normalized = str(backend).strip().lower()
    if normalized not in {"coolprop", "refprop"}:
        raise ValueError(f"Unsupported property backend: {backend!r}")

    _BACKEND = normalized
    _REFPROP_PATH = str(refprop_path or DEFAULT_REFPROP_PATH)
    if _BACKEND == "refprop":
        set_config_string(ALTERNATIVE_REFPROP_PATH, _REFPROP_PATH)
    _clear_property_caches()


def configure_property_backend_from_config(config: dict) -> None:
    fluids_cfg = config.get("fluids", {})
    backend = fluids_cfg.get("property_backend", fluids_cfg.get("backend", "coolprop"))
    refprop_path = fluids_cfg.get("refprop_path")
    configure_property_backend(str(backend), None if refprop_path is None else str(refprop_path))


def property_backend() -> str:
    return _BACKEND


def property_fluid_name(fluid: str) -> str:
    fluid_name = str(fluid)
    if _BACKEND != "refprop" or "::" in fluid_name:
        return fluid_name
    return f"REFPROP::{fluid_name}"


def props_si(output: str, input1: str, value1: float, input2: str, value2: float, fluid: str) -> float:
    return float(PropsSI(output, input1, value1, input2, value2, property_fluid_name(fluid)))


def fluid_property(output: str, fluid: str) -> float:
    return float(PropsSI(output, property_fluid_name(fluid)))


def _clear_property_caches() -> None:
    h_tp.cache_clear()
    p_sat.cache_clear()
    h_sat_liq.cache_clear()


@lru_cache(maxsize=65536)
def h_tp(temperature_k: float, pressure_pa: float, fluid: str) -> float:
    return props_si("H", "T", temperature_k, "P", pressure_pa, fluid)


@lru_cache(maxsize=65536)
def p_sat(temperature_k: float, fluid: str) -> float:
    return props_si("P", "T", temperature_k, "Q", 0.0, fluid)


@lru_cache(maxsize=65536)
def h_sat_liq(temperature_k: float, fluid: str) -> float:
    return props_si("H", "T", temperature_k, "Q", 0.0, fluid)


def h_refrigerant_liquid(temperature_k: float, pressure_pa: float, fluid: str, subcooling_k: float) -> float:
    if subcooling_k <= 1.0e-9:
        return h_sat_liq(temperature_k, fluid)
    return h_tp(temperature_k, pressure_pa, fluid)
