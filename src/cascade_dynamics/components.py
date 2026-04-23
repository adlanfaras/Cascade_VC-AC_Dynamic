from __future__ import annotations

import math


MIN_DT = 1.0e-6


def positive_lmtd(delta_t_1: float, delta_t_2: float) -> float:
    dt1 = max(delta_t_1, MIN_DT)
    dt2 = max(delta_t_2, MIN_DT)
    if abs(dt1 - dt2) < 1.0e-9:
        return 0.5 * (dt1 + dt2)
    return (dt1 - dt2) / math.log(dt1 / dt2)


def compressor_actual_enthalpy(h_in: float, h_out_is: float, eta_is: float) -> float:
    return h_in + (h_out_is - h_in) / eta_is


def turbine_actual_enthalpy(h_in: float, h_out_is: float, eta_is: float) -> float:
    return h_in - eta_is * (h_in - h_out_is)

