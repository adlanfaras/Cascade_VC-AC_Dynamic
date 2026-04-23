from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from .control import ControlSystem
from .model import CascadeSystemModel
from .numerics import newton_raphson_fd


def initial_vector(config: dict) -> np.ndarray:
    guess = config["initial_guess"]
    return np.array(
        [
            guess["room_c"],
            guess["sink_c"],
            guess["t3_c"],
            guess["t4_c"],
            guess["t6_c"],
            guess["tevap_c"],
            guess["m_ref_kg_s"],
        ],
        dtype=float,
    )


def solve_startup_initialization(config: dict, model: CascadeSystemModel) -> tuple[np.ndarray, int]:
    sim_cfg = config["simulation"]
    startup_cfg = sim_cfg.get("startup_initialization", {})
    time_s = startup_cfg.get("time_s", sim_cfg["t_start_s"])

    def residual_fn(x: np.ndarray) -> np.ndarray:
        return model.steady_state_residual(x, time_s)

    return newton_raphson_fd(
        residual_fn,
        initial_vector(config),
        tol=startup_cfg.get("newton_tol", sim_cfg["newton_tol"]),
        max_iter=startup_cfg.get("newton_max_iter", sim_cfg["newton_max_iter"]),
        step=startup_cfg.get("fd_step", sim_cfg["fd_step"]),
    )


def run_simulation(config: dict) -> list[dict[str, float]]:
    sim_cfg = config["simulation"]
    control = ControlSystem(config)
    model = CascadeSystemModel(config)

    dt_s = sim_cfg["dt_s"]
    times = np.arange(sim_cfg["t_start_s"], sim_cfg["t_end_s"] + dt_s, dt_s)
    startup_cfg = sim_cfg.get("startup_initialization", {})
    startup_enabled = startup_cfg.get("enabled", True)
    startup_iters = 0
    if startup_enabled:
        unknowns, startup_iters = solve_startup_initialization(config, model)
    else:
        unknowns = initial_vector(config)
    state = unknowns[:2].copy()
    history: list[dict[str, float]] = []

    for idx, time_s in enumerate(times):
        if idx == 0:
            step = model.post_process(unknowns, time_s)
            step.values["startup_initialization_enabled"] = float(startup_enabled)
            step.values["startup_newton_iterations"] = float(startup_iters)
            step.values["newton_iterations"] = 0.0
            history.append(step.values)
            continue

        prev_state = state.copy()
        controller_outputs = control.update(history[-1], config, dt_s)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            return model.residual(x, prev_state, time_s, dt_s)

        unknowns, iters = newton_raphson_fd(
            residual_fn,
            unknowns,
            tol=sim_cfg["newton_tol"],
            max_iter=sim_cfg["newton_max_iter"],
            step=sim_cfg["fd_step"],
        )
        step = model.post_process(unknowns, time_s)
        step.values["newton_iterations"] = float(iters)
        step.values.update(controller_outputs)
        history.append(step.values)
        state = step.state_vector

    return history


def save_plot(history: list[dict[str, float]], plot_file: str | Path) -> None:
    out_path = Path(plot_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_min = np.array([row["time_s"] for row in history]) / 60.0
    room_c = np.array([row["room_c"] for row in history])
    m_ref_kg_s = np.array([row["m_ref_kg_s"] for row in history])
    m_air_kg_s = np.array([row["m_air_kg_s"] for row in history])
    cop = np.array([row["cop_system"] for row in history])

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(t_min, room_c, label="Room")
    axes[0].set_ylabel("Temperature [C]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_min, m_ref_kg_s, label="Refrigerant mass flow")
    axes[1].set_ylabel("Mass Flow [kg/s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_min, m_air_kg_s, label="Air mass flow")
    axes[2].set_ylabel("Mass Flow [kg/s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_min, cop, label="COP")
    axes[3].set_xlabel("Time [min]")
    axes[3].set_ylabel("COP")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_csv(history: list[dict[str, float]], csv_file: str | Path) -> None:
    out_path = Path(csv_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
