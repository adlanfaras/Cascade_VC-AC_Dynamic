from __future__ import annotations

import math


MIN_DT = 1.0e-6


def positive_lmtd(delta_t_1: float, delta_t_2: float) -> float:
    dt1 = max(delta_t_1, MIN_DT)
    dt2 = max(delta_t_2, MIN_DT)
    if abs(dt1 - dt2) < 1.0e-9:
        return 0.5 * (dt1 + dt2)
    return (dt1 - dt2) / math.log(dt1 / dt2)


def single_stream_scaled_ua(
    ua_nominal_w_k: float,
    m_dot_kg_s: float,
    m_dot_nominal_kg_s: float,
    exponent: float,
    *,
    min_flow_ratio: float = 0.0,
    max_flow_ratio: float | None = None,
) -> float:
    ua_nominal = max(float(ua_nominal_w_k), 0.0)
    if ua_nominal <= 0.0:
        return 0.0

    m_dot_nominal = float(m_dot_nominal_kg_s)
    if m_dot_nominal <= 0.0:
        return ua_nominal

    flow_ratio = max(float(m_dot_kg_s), 0.0) / m_dot_nominal
    flow_ratio = max(flow_ratio, max(float(min_flow_ratio), 0.0))
    if max_flow_ratio is not None:
        flow_ratio = min(flow_ratio, max(float(max_flow_ratio), 0.0))
    return ua_nominal * flow_ratio ** float(exponent)


def compressor_actual_enthalpy(h_in: float, h_out_is: float, eta_is: float) -> float:
    return h_in + (h_out_is - h_in) / eta_is


def turbine_actual_enthalpy(h_in: float, h_out_is: float, eta_is: float) -> float:
    return h_in - eta_is * (h_in - h_out_is)

