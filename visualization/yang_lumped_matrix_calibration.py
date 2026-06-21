from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents, here.parent, *here.parents):
        if (candidate / "outputs").exists() and (candidate / "src").exists():
            return candidate
    return here.parents[1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cascade_dynamics.config import load_config  # noqa: E402
from src.cascade_dynamics.simulation import run_simulation, save_csv, save_plot  # noqa: E402
from visualization.yang_validation_plot import read_yang_trace, resolve_path  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a reduced-order matrix regenerator model to Yang test 1 traces.")
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "config" / "yang_validation.json"))
    parser.add_argument("--outlet-data", help="Yang expander outlet CSV.")
    parser.add_argument("--inlet-data", help="Yang expander inlet CSV.")
    parser.add_argument("--property-backend", choices=("coolprop", "refprop"), default="coolprop")
    parser.add_argument(
        "--model",
        choices=("lumped_matrix", "two_lump_matrix", "counterflow_cells"),
        default="counterflow_cells",
    )
    parser.add_argument("--cell-count", type=int, default=2, help="Cell count for counterflow_cells.")
    parser.add_argument("--fit-initial-solid", action="store_true", help="Also fit initial matrix temperature.")
    parser.add_argument("--initial-ua-w-k", type=float, default=220.0)
    parser.add_argument("--initial-solid-capacitance-j-k", type=float, default=10000.0)
    parser.add_argument("--initial-hot-solid-capacitance-j-k", type=float, default=5000.0)
    parser.add_argument("--initial-cold-solid-capacitance-j-k", type=float, default=5000.0)
    parser.add_argument("--initial-solid-k", type=float, default=300.0)
    parser.add_argument("--ua-min-w-k", type=float, default=10.0)
    parser.add_argument("--ua-max-w-k", type=float, default=5000.0)
    parser.add_argument("--solid-cap-min-j-k", type=float, default=100.0)
    parser.add_argument("--solid-cap-max-j-k", type=float, default=1000000.0)
    parser.add_argument("--initial-solid-min-k", type=float, default=285.0)
    parser.add_argument("--initial-solid-max-k", type=float, default=310.0)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--output-csv", default=str(PROJECT_ROOT / "outputs" / "yang_validation_counterflow_cells.csv"))
    parser.add_argument("--output-plot", default=str(PROJECT_ROOT / "outputs" / "yang_validation_counterflow_cells.png"))
    parser.add_argument("--fitted-config", default=str(PROJECT_ROOT / "config" / "yang_validation_counterflow_cells_fitted.json"))
    parser.add_argument("--summary", default=str(PROJECT_ROOT / "outputs" / "yang_validation_counterflow_cells_fit_summary.json"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def configure_lumped_matrix_case(
    base_config: dict[str, Any],
    *,
    model_name: str,
    cell_count: int,
    ua_w_k: float,
    solid_capacitance_j_k: float,
    hot_solid_capacitance_j_k: float | None,
    cold_solid_capacitance_j_k: float | None,
    initial_solid_k: float,
    output_csv: Path,
    output_plot: Path,
    property_backend: str,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("system", {})["mode"] = "air_cycle"
    config.setdefault("fluids", {})["property_backend"] = property_backend
    config["air_cycle"]["regenerator_ua_w_k"] = float(ua_w_k)
    reg_cfg = config["air_cycle"].setdefault("regenerator", {})
    if model_name == "counterflow_cells":
        reg_model = "transient_distributed"
        matrix_count = max(1, int(cell_count))
    else:
        reg_model = model_name
        matrix_count = 2 if model_name == "two_lump_matrix" else 1
    reg_cfg.update(
        {
            "model": reg_model,
            "cell_count": matrix_count,
            "ua_w_k": float(ua_w_k),
            "solid_capacitance_j_k": float(solid_capacitance_j_k),
            "initial_solid_k": float(initial_solid_k),
        }
    )
    if model_name == "two_lump_matrix":
        hot_cap = float(hot_solid_capacitance_j_k if hot_solid_capacitance_j_k is not None else 0.5 * solid_capacitance_j_k)
        cold_cap = float(cold_solid_capacitance_j_k if cold_solid_capacitance_j_k is not None else 0.5 * solid_capacitance_j_k)
        reg_cfg["hot_solid_capacitance_j_k"] = hot_cap
        reg_cfg["cold_solid_capacitance_j_k"] = cold_cap
        reg_cfg["solid_capacitance_j_k"] = hot_cap + cold_cap
    config.setdefault("output", {})["csv_file"] = str(output_csv)
    config.setdefault("output", {})["plot_file"] = str(output_plot)
    return config


def simulate(config: dict[str, Any], *, verbose: bool) -> pd.DataFrame:
    if verbose:
        history = run_simulation(config)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            history = run_simulation(config)
    model = pd.DataFrame(history)
    if "time_s" not in model or "t4_k" not in model or "t5_k" not in model:
        raise ValueError("Simulation did not produce time_s, t4_k, and t5_k columns.")
    model["time_min"] = model["time_s"] / 60.0
    return model


def trace_residuals(model: pd.DataFrame, outlet_exp: pd.DataFrame, inlet_exp: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    outlet_model = np.interp(
        outlet_exp["time_min"].to_numpy(),
        model["time_min"].to_numpy(),
        model["t5_k"].to_numpy(),
    )
    inlet_model = np.interp(
        inlet_exp["time_min"].to_numpy(),
        model["time_min"].to_numpy(),
        model["t4_k"].to_numpy(),
    )
    outlet_residual = outlet_model - outlet_exp["expander_outlet_experiment_k"].to_numpy()
    inlet_residual = inlet_model - inlet_exp["expander_inlet_experiment_k"].to_numpy()
    metrics = {
        "outlet_rmse_k": float(math.sqrt(np.mean(outlet_residual**2))),
        "inlet_rmse_k": float(math.sqrt(np.mean(inlet_residual**2))),
        "mean_rmse_k": float(0.5 * (math.sqrt(np.mean(outlet_residual**2)) + math.sqrt(np.mean(inlet_residual**2)))),
        "outlet_mae_k": float(np.mean(np.abs(outlet_residual))),
        "inlet_mae_k": float(np.mean(np.abs(inlet_residual))),
        "outlet_mean_error_k": float(np.mean(outlet_residual)),
        "inlet_mean_error_k": float(np.mean(inlet_residual)),
        "outlet_final_error_k": float(outlet_residual[-1]),
        "inlet_final_error_k": float(inlet_residual[-1]),
    }
    return np.concatenate([outlet_residual, inlet_residual]), metrics


def main() -> None:
    args = build_arg_parser().parse_args()
    base_config = load_config(args.base_config)
    outlet_path = resolve_path(
        args.outlet_data,
        [
            PROJECT_ROOT / "Data_Test1_yang.csv",
            PROJECT_ROOT.parent / "Data_Test1_yang.csv",
            Path.home() / "Downloads" / "Data_Test1_yang.csv",
        ],
        "Yang expander outlet CSV",
    )
    inlet_path = resolve_path(
        args.inlet_data,
        [
            PROJECT_ROOT / "Data_Test1_yang_inlet.csv",
            PROJECT_ROOT.parent / "Data_Test1_yang_inlet.csv",
            Path.home() / "Downloads" / "Data_Test1_yang_inlet.csv",
        ],
        "Yang expander inlet CSV",
    )
    outlet_exp = read_yang_trace(outlet_path, "expander_outlet")
    inlet_exp = read_yang_trace(inlet_path, "expander_inlet")

    output_csv = Path(args.output_csv).expanduser()
    output_plot = Path(args.output_plot).expanduser()
    fitted_config_path = Path(args.fitted_config).expanduser()
    summary_path = Path(args.summary).expanduser()

    def decode(x: np.ndarray) -> tuple[float, float, float | None, float | None, float]:
        ua_w_k = float(np.exp(x[0]))
        if args.model == "two_lump_matrix":
            hot_cap = float(np.exp(x[1]))
            cold_cap = float(np.exp(x[2]))
            solid_capacitance_j_k = hot_cap + cold_cap
            initial_idx = 3
        else:
            solid_capacitance_j_k = float(np.exp(x[1]))
            hot_cap = None
            cold_cap = None
            initial_idx = 2
        initial_solid_k = float(x[initial_idx]) if args.fit_initial_solid else float(args.initial_solid_k)
        return ua_w_k, solid_capacitance_j_k, hot_cap, cold_cap, initial_solid_k

    evaluation_count = 0

    def residual_fn(x: np.ndarray) -> np.ndarray:
        nonlocal evaluation_count
        evaluation_count += 1
        ua_w_k, solid_capacitance_j_k, hot_cap, cold_cap, initial_solid_k = decode(x)
        config = configure_lumped_matrix_case(
            base_config,
            model_name=args.model,
            cell_count=args.cell_count,
            ua_w_k=ua_w_k,
            solid_capacitance_j_k=solid_capacitance_j_k,
            hot_solid_capacitance_j_k=hot_cap,
            cold_solid_capacitance_j_k=cold_cap,
            initial_solid_k=initial_solid_k,
            output_csv=output_csv,
            output_plot=output_plot,
            property_backend=args.property_backend,
        )
        try:
            model = simulate(config, verbose=False)
            residual, metrics = trace_residuals(model, outlet_exp, inlet_exp)
        except Exception as exc:
            if args.verbose:
                print(f"[fit] failed at UA={ua_w_k:.6g}, C={solid_capacitance_j_k:.6g}: {exc}")
            return np.full(len(outlet_exp) + len(inlet_exp), 1.0e6, dtype=float)
        if args.verbose:
            print(
                "[fit] "
                f"eval={evaluation_count:03d} UA={ua_w_k:.4g} W/K "
                f"C={solid_capacitance_j_k:.4g} J/K "
                f"T0={initial_solid_k:.2f} K mean_RMSE={metrics['mean_rmse_k']:.4f} K"
            )
        return residual

    x0 = [math.log(args.initial_ua_w_k)]
    lower = [math.log(args.ua_min_w_k)]
    upper = [math.log(args.ua_max_w_k)]
    if args.model == "two_lump_matrix":
        x0.extend([math.log(args.initial_hot_solid_capacitance_j_k), math.log(args.initial_cold_solid_capacitance_j_k)])
        lower.extend([math.log(args.solid_cap_min_j_k), math.log(args.solid_cap_min_j_k)])
        upper.extend([math.log(args.solid_cap_max_j_k), math.log(args.solid_cap_max_j_k)])
    else:
        x0.append(math.log(args.initial_solid_capacitance_j_k))
        lower.append(math.log(args.solid_cap_min_j_k))
        upper.append(math.log(args.solid_cap_max_j_k))
    if args.fit_initial_solid:
        x0.append(args.initial_solid_k)
        lower.append(args.initial_solid_min_k)
        upper.append(args.initial_solid_max_k)

    result = least_squares(
        residual_fn,
        np.asarray(x0, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        loss="soft_l1",
        f_scale=5.0,
        max_nfev=args.max_nfev,
        x_scale="jac",
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
    )
    (
        fitted_ua_w_k,
        fitted_solid_capacitance_j_k,
        fitted_hot_solid_capacitance_j_k,
        fitted_cold_solid_capacitance_j_k,
        fitted_initial_solid_k,
    ) = decode(np.asarray(result.x, dtype=float))
    fitted_config = configure_lumped_matrix_case(
        base_config,
        model_name=args.model,
        cell_count=args.cell_count,
        ua_w_k=fitted_ua_w_k,
        solid_capacitance_j_k=fitted_solid_capacitance_j_k,
        hot_solid_capacitance_j_k=fitted_hot_solid_capacitance_j_k,
        cold_solid_capacitance_j_k=fitted_cold_solid_capacitance_j_k,
        initial_solid_k=fitted_initial_solid_k,
        output_csv=output_csv,
        output_plot=output_plot,
        property_backend=args.property_backend,
    )
    final_history = run_simulation(fitted_config) if args.verbose else None
    if final_history is None:
        with contextlib.redirect_stdout(io.StringIO()):
            final_history = run_simulation(fitted_config)
    save_csv(final_history, output_csv)
    save_plot(final_history, output_plot)
    final_model = pd.DataFrame(final_history)
    final_model["time_min"] = final_model["time_s"] / 60.0
    final_residual, final_metrics = trace_residuals(final_model, outlet_exp, inlet_exp)

    fitted_config.setdefault("validation_reference", {})
    fitted_config["validation_reference"].update(
        {
            "validation_temperature_source": "two_trace_t4_t5",
            f"{args.model}_fit_to_yang_data": True,
            f"{args.model}_fit_outlet_rmse_k": final_metrics["outlet_rmse_k"],
            f"{args.model}_fit_inlet_rmse_k": final_metrics["inlet_rmse_k"],
            f"{args.model}_fit_mean_rmse_k": final_metrics["mean_rmse_k"],
        }
    )

    fitted_config_path.parent.mkdir(parents=True, exist_ok=True)
    fitted_config_path.write_text(json.dumps(fitted_config, indent=2), encoding="utf-8")
    summary = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "model": args.model,
        "fit_initial_solid": bool(args.fit_initial_solid),
        "fitted": {
            "ua_w_k": fitted_ua_w_k,
            "solid_capacitance_j_k": fitted_solid_capacitance_j_k,
            "hot_solid_capacitance_j_k": fitted_hot_solid_capacitance_j_k,
            "cold_solid_capacitance_j_k": fitted_cold_solid_capacitance_j_k,
            "initial_solid_k": fitted_initial_solid_k,
        },
        "metrics": final_metrics,
        "residual_norm_k": float(np.linalg.norm(final_residual)),
        "files": {
            "fitted_config": str(fitted_config_path),
            "output_csv": str(output_csv),
            "output_plot": str(output_plot),
            "outlet_data": str(outlet_path),
            "inlet_data": str(inlet_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
