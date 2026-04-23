from __future__ import annotations

from typing import Any


GRAVITY_M_S2 = 9.80665


def mass_flow_from_isentropic_head(
    config: dict[str, Any],
    head_j_kg: float,
    suction_density_kg_m3: float | None = None,
) -> float:
    if suction_density_kg_m3 is None:
        raise ValueError("suction_density_kg_m3 is required for volumetric compressor flow maps.")

    head_m = head_j_kg / GRAVITY_M_S2
    head_min_m = float(config["head_min_m"])
    head_max_m = float(config["head_max_m"])
    head_eval_m = min(max(head_m, head_min_m), head_max_m)

    if "coefficients" in config:
        coefficients = [float(value) for value in config["coefficients"]]
        volumetric_flow_m3_s = 0.0
        for coefficient in coefficients:
            volumetric_flow_m3_s = volumetric_flow_m3_s * head_eval_m + coefficient
    else:
        a = float(config["a"])
        b = float(config["b"])
        c = float(config["c"])
        d = float(config["d"])
        volumetric_flow_m3_s = ((a * head_eval_m + b) * head_eval_m + c) * head_eval_m + d
    volumetric_flow_m3_s *= float(config.get("speed_fraction", 1.0))
    volumetric_flow_m3_s = max(float(config.get("q_min_m3_s", 0.0)), volumetric_flow_m3_s)
    volumetric_flow_m3_s = min(float(config.get("q_max_m3_s", 1.0e9)), volumetric_flow_m3_s)

    mass_flow = volumetric_flow_m3_s * suction_density_kg_m3
    mass_flow = max(float(config.get("m_dot_min_kg_s", 0.0)), mass_flow)
    mass_flow = min(float(config.get("m_dot_max_kg_s", 1.0e9)), mass_flow)
    return mass_flow
