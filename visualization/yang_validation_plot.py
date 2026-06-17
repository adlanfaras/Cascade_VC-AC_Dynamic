from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents, here.parent, *here.parents):
        if (candidate / "outputs").exists() and (candidate / "src").exists():
            return candidate
    return here.parents[1]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 18,
            "axes.linewidth": 1.5,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.grid": False,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(which="major", direction="in", top=True, right=True, length=7, width=1.3)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3.5, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def resolve_path(value: str | None, candidates: list[Path], label: str) -> Path:
    if value:
        path = Path(value).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"{label} not found: {path}")

    for path in candidates:
        if path.exists():
            return path
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} not found. Searched:\n  {searched}")


def read_yang_trace(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        header=None,
        names=["time_min", f"{label}_experiment_k"],
        engine="python",
    )
    df["time_min"] = pd.to_numeric(df["time_min"], errors="coerce")
    df[f"{label}_experiment_k"] = pd.to_numeric(df[f"{label}_experiment_k"], errors="coerce")
    return df.dropna().sort_values("time_min").reset_index(drop=True)


def read_model(path: Path) -> pd.DataFrame:
    model = pd.read_csv(path)
    required = {"time_s", "t4_k", "t5_k"}
    missing = sorted(required - set(model.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    model = model.copy()
    model["time_min"] = model["time_s"] / 60.0
    return model


def compare_trace(
    experiment: pd.DataFrame,
    model: pd.DataFrame,
    *,
    label: str,
    model_column: str,
    experiment_path: Path,
    model_path: Path,
) -> tuple[pd.DataFrame, dict[str, float | str | int]]:
    experiment_col = f"{label}_experiment_k"
    model_col = f"{label}_model_k"
    residual_col = f"{label}_residual_model_minus_experiment_k"
    model_values = np.interp(
        experiment["time_min"].to_numpy(),
        model["time_min"].to_numpy(),
        model[model_column].to_numpy(),
    )

    result = experiment.copy()
    result[model_col] = model_values
    result[residual_col] = result[model_col] - result[experiment_col]
    residual = result[residual_col].to_numpy()

    summary: dict[str, float | str | int] = {
        "trace": label,
        "experiment_file": str(experiment_path),
        "model_file": str(model_path),
        "model_column": model_column,
        "n_points": int(len(result)),
        "time_min_first": float(result["time_min"].iloc[0]),
        "time_min_last": float(result["time_min"].iloc[-1]),
        "rmse_k": float(math.sqrt(np.mean(residual**2))),
        "mae_k": float(np.mean(np.abs(residual))),
        "mean_error_k": float(np.mean(residual)),
        "max_abs_error_k": float(np.max(np.abs(residual))),
        "final_experiment_k": float(result[experiment_col].iloc[-1]),
        "final_model_k": float(result[model_col].iloc[-1]),
        "final_error_k": float(residual[-1]),
    }
    return result, summary


def save_single_trace_plot(
    model: pd.DataFrame,
    result: pd.DataFrame,
    summary: dict[str, float | str | int],
    *,
    label: str,
    model_column: str,
    output_path: Path,
) -> None:
    experiment_col = f"{label}_experiment_k"
    residual_col = f"{label}_residual_model_minus_experiment_k"
    title = "Expander outlet temperature" if label == "expander_outlet" else "Expander inlet temperature"
    model_color = OKABE_ITO["blue"] if label == "expander_outlet" else OKABE_ITO["bluish_green"]
    data_color = OKABE_ITO["vermillion"] if label == "expander_outlet" else OKABE_ITO["orange"]
    marker = "o" if label == "expander_outlet" else "s"

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
    )

    axes[0].plot(model["time_min"], model[model_column], color=model_color, linewidth=2.4, label=f"Model {model_column}")
    axes[0].scatter(
        result["time_min"],
        result[experiment_col],
        color=data_color,
        marker=marker,
        s=38,
        label="Yang test 1 data",
        zorder=3,
    )
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title(title, loc="left")
    axes[0].legend(frameon=False, loc="best")
    axes[0].text(
        0.02,
        0.04,
        (
            f"RMSE {summary['rmse_k']:.2f} K | "
            f"MAE {summary['mae_k']:.2f} K | "
            f"final error {summary['final_error_k']:+.2f} K"
        ),
        transform=axes[0].transAxes,
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "0.35", "alpha": 0.9, "boxstyle": "square,pad=0.3"},
    )

    axes[1].axhline(0.0, color=OKABE_ITO["black"], linewidth=1.2)
    axes[1].plot(result["time_min"], result[residual_col], color=model_color, linewidth=2.1)
    axes[1].set_ylabel("Residual (K)")
    axes[1].set_xlabel("Time (min)")

    for ax in axes:
        style_axes(ax)

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_two_trace_plot(
    model: pd.DataFrame,
    outlet: pd.DataFrame,
    inlet: pd.DataFrame,
    outlet_summary: dict[str, float | str | int],
    inlet_summary: dict[str, float | str | int],
    output_path: Path,
) -> None:
    score = 0.5 * (float(outlet_summary["rmse_k"]) + float(inlet_summary["rmse_k"]))
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]},
    )

    axes[0].plot(model["time_min"], model["t5_k"], color=OKABE_ITO["blue"], linewidth=2.5, label="Model outlet, $T_5$")
    axes[0].scatter(
        outlet["time_min"],
        outlet["expander_outlet_experiment_k"],
        color=OKABE_ITO["vermillion"],
        marker="o",
        s=36,
        label="Yang outlet data",
        zorder=3,
    )
    axes[0].plot(
        model["time_min"],
        model["t4_k"],
        color=OKABE_ITO["bluish_green"],
        linewidth=2.5,
        linestyle="--",
        label="Model inlet, $T_4$",
    )
    axes[0].scatter(
        inlet["time_min"],
        inlet["expander_inlet_experiment_k"],
        color=OKABE_ITO["orange"],
        marker="s",
        s=34,
        label="Yang inlet data",
        zorder=3,
    )
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("Yang test 1 validation: expander inlet and outlet", loc="left")
    axes[0].legend(frameon=False, loc="best", ncols=2)
    axes[0].text(
        0.02,
        0.04,
        (
            f"Outlet RMSE {outlet_summary['rmse_k']:.2f} K | "
            f"Inlet RMSE {inlet_summary['rmse_k']:.2f} K | "
            f"Mean RMSE {score:.2f} K"
        ),
        transform=axes[0].transAxes,
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "0.35", "alpha": 0.9, "boxstyle": "square,pad=0.3"},
    )

    axes[1].axhline(0.0, color=OKABE_ITO["black"], linewidth=1.2)
    axes[1].plot(
        outlet["time_min"],
        outlet["expander_outlet_residual_model_minus_experiment_k"],
        color=OKABE_ITO["blue"],
        linewidth=2.2,
        label="Outlet residual",
    )
    axes[1].plot(
        inlet["time_min"],
        inlet["expander_inlet_residual_model_minus_experiment_k"],
        color=OKABE_ITO["bluish_green"],
        linewidth=2.2,
        linestyle="--",
        label="Inlet residual",
    )
    axes[1].set_ylabel("Model - data (K)")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend(frameon=False, loc="best")

    for ax in axes:
        style_axes(ax)

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Yang test 1 validation plots and residual summaries.")
    parser.add_argument("--model-csv", help="Simulation CSV with t4_k and t5_k columns.")
    parser.add_argument("--outlet-data", help="Yang expander outlet CSV, semicolon-delimited with decimal commas.")
    parser.add_argument("--inlet-data", help="Yang expander inlet CSV, semicolon-delimited with decimal commas.")
    parser.add_argument("--output-dir", default=str(project_root / "outputs"), help="Directory for generated plots and summaries.")
    parser.add_argument("--combined-name", default="yang_validation_two_trace_fit.png", help="Combined plot filename.")
    return parser


def main() -> None:
    project_root = find_project_root()
    parser = build_arg_parser(project_root)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(args.model_csv, [project_root / "outputs" / "yang_validation.csv"], "model CSV")
    outlet_path = resolve_path(
        args.outlet_data,
        [
            project_root / "Data_Test1_yang.csv",
            project_root.parent / "Data_Test1_yang.csv",
            Path.home() / "Downloads" / "Data_Test1_yang.csv",
        ],
        "Yang expander outlet CSV",
    )
    inlet_path = resolve_path(
        args.inlet_data,
        [
            project_root / "Data_Test1_yang_inlet.csv",
            project_root.parent / "Data_Test1_yang_inlet.csv",
            Path.home() / "Downloads" / "Data_Test1_yang_inlet.csv",
        ],
        "Yang expander inlet CSV",
    )

    configure_plot_style()
    model = read_model(model_path)
    outlet_exp = read_yang_trace(outlet_path, "expander_outlet")
    inlet_exp = read_yang_trace(inlet_path, "expander_inlet")

    outlet, outlet_summary = compare_trace(
        outlet_exp,
        model,
        label="expander_outlet",
        model_column="t5_k",
        experiment_path=outlet_path,
        model_path=model_path,
    )
    inlet, inlet_summary = compare_trace(
        inlet_exp,
        model,
        label="expander_inlet",
        model_column="t4_k",
        experiment_path=inlet_path,
        model_path=model_path,
    )

    outlet.to_csv(output_dir / "yang_validation_outlet_vs_test1_residuals.csv", index=False)
    inlet.to_csv(output_dir / "yang_validation_inlet_vs_test1_residuals.csv", index=False)
    (output_dir / "yang_validation_outlet_vs_test1_summary.json").write_text(
        json.dumps(outlet_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "yang_validation_inlet_vs_test1_summary.json").write_text(
        json.dumps(inlet_summary, indent=2),
        encoding="utf-8",
    )

    combined_summary = {
        "mapping": {
            outlet_path.name: "expander_outlet -> model t5_k",
            inlet_path.name: "expander_inlet -> model t4_k",
        },
        "combined_score_mean_rmse_k": 0.5 * (float(outlet_summary["rmse_k"]) + float(inlet_summary["rmse_k"])),
        "outlet": outlet_summary,
        "inlet": inlet_summary,
    }
    (output_dir / "yang_validation_two_trace_fit_summary.json").write_text(
        json.dumps(combined_summary, indent=2),
        encoding="utf-8",
    )

    save_single_trace_plot(
        model,
        outlet,
        outlet_summary,
        label="expander_outlet",
        model_column="t5_k",
        output_path=output_dir / "yang_validation_outlet_vs_test1.png",
    )
    save_single_trace_plot(
        model,
        inlet,
        inlet_summary,
        label="expander_inlet",
        model_column="t4_k",
        output_path=output_dir / "yang_validation_inlet_vs_test1.png",
    )
    save_two_trace_plot(
        model,
        outlet,
        inlet,
        outlet_summary,
        inlet_summary,
        output_path=output_dir / args.combined_name,
    )

    print(json.dumps(combined_summary, indent=2))


if __name__ == "__main__":
    main()
