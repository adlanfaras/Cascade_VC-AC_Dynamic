from __future__ import annotations

from typing import Any

import numpy as np


GRAVITY_M_S2 = 9.80665
AMMONIA_MAP_BASE_RPM = 1533.0
AMMONIA_MAP_VALIDITY = {
    "to_K": (248.45, 268.30),
    "tc_K": (283.15, 319.85),
    "N_rpm": (1226.0, 1610.0),
}
AMMONIA_MAP_COEFFS = {
    "Q_W": np.array(
        [
            [178291.032505071, 157914.917002567, -21094.3604156214, -3.40895823435749e-08],
            [7866.39033285544, 6967.37440503282, -930.705658717617, -8.56381665530458e-09],
            [-1286.05984566415, -1139.08159560756, 152.159138459414, 1.61967518688327e-08],
            [120.15572899345, 106.42364736346, -14.2161286395481, 1.88499465715957e-11],
            [-44.9871494248319, -39.8457615494645, 5.3226184779154, 7.24022555005662e-10],
            [7.38685440697709, 6.54264257825163, -0.873969750584109, -3.53871545534335e-10],
            [0.726432917975249, 0.643412023242175, -0.085947327656649, 1.19029874006476e-11],
            [-0.590995767061506, -0.523453401963714, 0.0699231898524894, 1.44474504632549e-11],
            [0.378568699641861, 0.335303710701287, -0.0447900518631015, -9.77068313248619e-12],
            [-0.0717177709681818, -0.0635214552909869, 0.00848523051243049, 3.32475717697413e-12],
        ],
        dtype=float,
    ),
    "P_W": np.array(
        [
            [6690.9237260078, 6530.28901958098, -1123.00482311539, 2261.11192179499],
            [-551.786968776613, -538.539748905394, 92.6119400908899, -186.469334355656],
            [375.260083545608, 366.250895005601, -62.9836627951941, 126.814335870486],
            [-15.7006134408586, -15.3236754373643, 2.63519139387988, -5.30582109215716],
            [13.9679806594049, 13.6326394472649, -2.34438625993174, 4.72030004929749],
            [12.0343644800643, 11.7454452389489, -2.0198480669399, 4.06685924279109],
            [-0.0811892551031285, -0.0792400754841754, 0.0136268068203364, -0.0274368682362421],
            [-0.0327692605628038, -0.0319825410057347, 0.0054999935992554, -0.0110739516298171],
            [0.197304258176268, 0.192567406750861, -0.0331155521462164, 0.0666764453820828],
            [-0.168743943726166, -0.164692764102592, 0.0283219881791477, -0.057024853146524],
        ],
        dtype=float,
    ),
    "mdot_kg_s": np.array(
        [
            [0.141303928587793, 0.125154910157601, -0.0167182609012475, -8.57189832069925e-14],
            [0.00640463313524958, 0.00567267515238645, -0.000757759029083879, -2.51175530036627e-14],
            [-0.000538716824273169, -0.000477149194761435, 6.37378486937534e-05, 3.91932891924745e-15],
            [0.000104473772219973, 9.25339140014983e-05, -1.23607305102497e-05, -2.24166165039057e-16],
            [-1.5929581682287e-05, -1.41090582846455e-05, 1.88469567179597e-06, 2.65926862575398e-16],
            [6.06439107646192e-06, 5.37131789554971e-06, -7.1750356298348e-07, -1.14051595805427e-16],
            [7.27923212493189e-07, 6.44731998407379e-07, -8.61236506593641e-08, -7.54513245679893e-20],
            [-2.12139636160394e-07, -1.87895109285788e-07, 2.5099130790971e-08, 3.88535526901849e-18],
            [3.55689253582464e-07, 3.15039058156678e-07, -4.2083088564111e-08, -3.65305347799972e-18],
            [-7.33312309147692e-08, -6.49505198376801e-08, 8.67612573063924e-09, 1.11345897221083e-18],
        ],
        dtype=float,
    ),
}
AIR_SPEED_MAP_COEFFS = np.array(
    [
        [138.35264089, -3.12206895, 26.94871441],
        [-463.65671896, 106.99401466, -125.26043030],
        [637.11935763, -285.93949983, 246.31005689],
        [-436.34528449, 299.16922017, -243.12916222],
        [148.88376117, -141.15631829, 117.80846616],
        [-20.26662640, 25.21224461, -22.29194904],
    ],
    dtype=float,
)


def _ammonia_map_basis(to_c: np.ndarray, tc_c: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.ones_like(to_c, dtype=float),
            to_c,
            tc_c,
            to_c**2,
            to_c * tc_c,
            tc_c**2,
            to_c**3,
            tc_c * to_c**2,
            to_c * tc_c**2,
            tc_c**3,
        ],
        axis=-1,
    )


def ammonia_compressor_map(tc_k: float, to_k: float, speed_rpm: float, check_range: bool = True) -> dict[str, float]:
    tc_arr, to_arr, speed_arr = np.broadcast_arrays(
        np.asarray(tc_k, dtype=float),
        np.asarray(to_k, dtype=float),
        np.asarray(speed_rpm, dtype=float),
    )
    if check_range:
        for key, values in (
            ("to_K", to_arr),
            ("tc_K", tc_arr),
            ("N_rpm", speed_arr),
        ):
            lo, hi = AMMONIA_MAP_VALIDITY[key]
            if np.any((values < lo) | (values > hi)):
                raise ValueError(f"{key} outside valid range {lo} to {hi}")

    to_c = to_arr - 273.15
    tc_c = tc_arr - 273.15
    x = (speed_arr - AMMONIA_MAP_BASE_RPM) / AMMONIA_MAP_BASE_RPM
    basis = _ammonia_map_basis(to_c, tc_c)
    speed_terms = np.stack([np.ones_like(x), x, x**2, x**3], axis=-1)

    outputs: dict[str, float] = {}
    for name, coeffs in AMMONIA_MAP_COEFFS.items():
        c = np.einsum("ij,...j->...i", coeffs, speed_terms)
        outputs[name] = float(np.einsum("...i,...i->...", c, basis))
    outputs["COP"] = outputs["Q_W"] / max(outputs["P_W"], 1.0e-9)
    return outputs


def volumetric_flow_from_head(config: dict[str, Any], head_m: float) -> float:
    model = config.get("model", "polynomial_volumetric_flow_head")
    head_min_m = float(config.get("head_min_m", head_m))
    head_max_m = float(config.get("head_max_m", head_m))
    head_eval_m = min(max(head_m, head_min_m), head_max_m)

    if model in {"polynomial_volumetric_flow_head_speed", "polynomial_volumetric_flow_head_speed_constant_mass_flow"}:
        h = head_eval_m / 1000.0
        rpm = float(config["speed_rpm"])
        design_speed_rpm = float(config.get("design_speed_rpm", 15000.0))
        speed_scale_rpm = float(config.get("speed_scale_rpm", 5000.0))
        u = (rpm - design_speed_rpm) / speed_scale_rpm
        volumetric_flow_m3_s = 0.0
        for i in range(AIR_SPEED_MAP_COEFFS.shape[0]):
            coeff_i = (
                AIR_SPEED_MAP_COEFFS[i, 0]
                + AIR_SPEED_MAP_COEFFS[i, 1] * u
                + AIR_SPEED_MAP_COEFFS[i, 2] * u**2
            )
            volumetric_flow_m3_s += coeff_i * h**i
    else:
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
    return volumetric_flow_m3_s


def _speed_map_head_from_volumetric_flow(config: dict[str, Any], target_q_m3_s: float) -> float | None:
    head_min_m = float(config["head_min_m"])
    head_max_m = float(config["head_max_m"])
    q_min_m3_s = float(config.get("q_min_m3_s", 0.0))
    q_max_m3_s = float(config.get("q_max_m3_s", 1.0e9))
    target_q = min(max(float(target_q_m3_s), q_min_m3_s), q_max_m3_s)

    rpm = float(config["speed_rpm"])
    design_speed_rpm = float(config.get("design_speed_rpm", 15000.0))
    speed_scale_rpm = float(config.get("speed_scale_rpm", 5000.0))
    u = (rpm - design_speed_rpm) / speed_scale_rpm

    coeffs_ascending = np.array(
        [
            AIR_SPEED_MAP_COEFFS[i, 0]
            + AIR_SPEED_MAP_COEFFS[i, 1] * u
            + AIR_SPEED_MAP_COEFFS[i, 2] * u**2
            for i in range(AIR_SPEED_MAP_COEFFS.shape[0])
        ],
        dtype=float,
    )
    coeffs_ascending[0] -= target_q
    roots = np.roots(coeffs_ascending[::-1])

    h_min = head_min_m / 1000.0
    h_max = head_max_m / 1000.0
    candidates = [
        float(root.real)
        for root in roots
        if abs(root.imag) <= 1.0e-7 and h_min - 1.0e-9 <= root.real <= h_max + 1.0e-9
    ]
    if not candidates:
        return None

    def flow_error(h: float) -> float:
        return abs(volumetric_flow_from_head(config, 1000.0 * h) - target_q)

    h_solution = min(candidates, key=flow_error)
    return float(np.clip(1000.0 * h_solution, head_min_m, head_max_m))


def mass_flow_from_isentropic_head(
    config: dict[str, Any],
    head_j_kg: float,
    suction_density_kg_m3: float | None = None,
) -> float:
    if suction_density_kg_m3 is None:
        raise ValueError("suction_density_kg_m3 is required for volumetric compressor flow maps.")

    head_m = head_j_kg / GRAVITY_M_S2
    volumetric_flow_m3_s = volumetric_flow_from_head(config, head_m)
    mass_flow = volumetric_flow_m3_s * suction_density_kg_m3
    mass_flow = max(float(config.get("m_dot_min_kg_s", 0.0)), mass_flow)
    mass_flow = min(float(config.get("m_dot_max_kg_s", 1.0e9)), mass_flow)
    return mass_flow


def head_from_mass_flow(
    config: dict[str, Any],
    target_m_dot_kg_s: float,
    suction_density_kg_m3: float,
) -> tuple[float, float]:
    if suction_density_kg_m3 <= 0.0:
        raise ValueError("suction_density_kg_m3 must be positive.")

    head_min_m = float(config["head_min_m"])
    head_max_m = float(config["head_max_m"])
    target_m_dot = float(target_m_dot_kg_s)
    model = config.get("model", "polynomial_volumetric_flow_head")

    if model in {"polynomial_volumetric_flow_head_speed", "polynomial_volumetric_flow_head_speed_constant_mass_flow"}:
        target_q_m3_s = target_m_dot / suction_density_kg_m3
        head = _speed_map_head_from_volumetric_flow(config, target_q_m3_s)
        if head is not None:
            return head, volumetric_flow_from_head(config, head) * suction_density_kg_m3

    def residual(head_m: float) -> float:
        return volumetric_flow_from_head(config, head_m) * suction_density_kg_m3 - target_m_dot

    bracket: tuple[float, float] | None = None
    f_min = residual(head_min_m)
    f_max = residual(head_max_m)
    if f_min == 0.0:
        return head_min_m, volumetric_flow_from_head(config, head_min_m) * suction_density_kg_m3
    if f_max == 0.0:
        return head_max_m, volumetric_flow_from_head(config, head_max_m) * suction_density_kg_m3
    if f_min * f_max <= 0.0:
        bracket = (head_min_m, head_max_m)

    if bracket is None:
        sample_heads = np.linspace(head_min_m, head_max_m, 200)
        sample_residuals = np.array([residual(float(head_m)) for head_m in sample_heads], dtype=float)
        best_idx = int(np.argmin(np.abs(sample_residuals)))

        for idx in range(len(sample_heads) - 1):
            f_lo = sample_residuals[idx]
            f_hi = sample_residuals[idx + 1]
            if f_lo == 0.0:
                head = float(sample_heads[idx])
                return head, volumetric_flow_from_head(config, head) * suction_density_kg_m3
            if f_lo * f_hi <= 0.0:
                bracket = (float(sample_heads[idx]), float(sample_heads[idx + 1]))
                break

        if bracket is None:
            head = float(sample_heads[best_idx])
            return head, volumetric_flow_from_head(config, head) * suction_density_kg_m3

    lo, hi = bracket
    f_lo = residual(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) <= 1.0e-9 or abs(hi - lo) <= 1.0e-7:
            return mid, volumetric_flow_from_head(config, mid) * suction_density_kg_m3
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    head = 0.5 * (lo + hi)
    return head, volumetric_flow_from_head(config, head) * suction_density_kg_m3
