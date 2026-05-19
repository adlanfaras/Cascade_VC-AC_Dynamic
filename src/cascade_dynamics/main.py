from __future__ import annotations

import argparse
import os
from pathlib import Path

from .batch import CaseResult, apply_output_version, run_case_from_config_path, run_cases
from .config import load_config
from .simulation import run_simulation, save_csv, save_plot


def _print_final_state(last: dict[str, float], plot_file: str | Path, csv_file: str | Path) -> None:
    print("Final dynamic state")
    print(f"  Room temperature     : {last['room_c']:.2f} C")
    print(f"  Dock temperature     : {last['dock_c']:.2f} C")
    print(f"  Sink temperature     : {last['sink_c']:.2f} C")
    print(f"  Cooling capacity     : {last['q_room_w'] / 1000.0:.2f} kW")
    print(f"  Dock evaporator duty : {last['q_dock_w'] / 1000.0:.2f} kW")
    print(f"  Useful cooling total : {last['q_useful_w'] / 1000.0:.2f} kW")
    print(f"  Cascade duty         : {last['q_cascade_w'] / 1000.0:.2f} kW")
    print(f"  Condenser duty       : {last['q_cond_w'] / 1000.0:.2f} kW")
    print(f"  NH3 compressor work  : {last['w_ref_comp_w'] / 1000.0:.2f} kW")
    print(f"  NH3 superheat        : {last['refrigerant_superheat_k']:.2f} K")
    print(f"  NH3 isentropic work  : {last['w_ref_isentropic_w'] / 1000.0:.2f} kW")
    print(f"  Air input power      : {last['w_air_input_w'] / 1000.0:.2f} kW")
    print(f"  System COP           : {last['cop_system']:.3f}")
    print(f"  Room-only COP        : {last['cop_room_only']:.3f}")
    print(f"  Dry-air massflow     : {last['m_air_kg_s']:.4f} kg/s")
    print(f"  Ice at air outlet    : {last['ice_mass_flow_kg_s']:.4f} kg/s")
    print(f"  Refrigerant massflow : {last['m_ref_kg_s']:.4f} kg/s")
    print(f"  Plot saved to        : {Path(plot_file).resolve()}")
    print(f"  CSV saved to         : {Path(csv_file).resolve()}")


def _print_case_result(result: CaseResult) -> None:
    duration = "base" if result.door_open_duration_s is None else f"{result.door_open_duration_s:g} s"
    print(f"[{result.name}] door open duration: {duration}")
    print(f"  Room temperature     : {result.final_room_c:.2f} C")
    print(f"  Dock temperature     : {result.final_dock_c:.2f} C")
    print(f"  Sink temperature     : {result.final_sink_c:.2f} C")
    print(f"  System COP           : {result.final_cop_system:.3f}")
    print(f"  Dry-air massflow     : {result.final_m_air_kg_s:.4f} kg/s")
    print(f"  Refrigerant massflow : {result.final_m_ref_kg_s:.4f} kg/s")
    print(f"  Plot saved to        : {result.plot_file}")
    print(f"  CSV saved to         : {result.csv_file}")


def _door_durations_from_args(args: argparse.Namespace) -> list[float] | None:
    durations: list[float] = []
    if args.door_open_duration is not None:
        durations.extend(args.door_open_duration)
    if args.door_open_durations is not None:
        durations.extend(args.door_open_durations)
    return durations or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic simulation of a cascade reverse-Brayton / VCC refrigeration system.")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file.")
    parser.add_argument(
        "--door-open-duration",
        type=float,
        action="append",
        help="Run one case with this door open duration in seconds. Can be repeated.",
    )
    parser.add_argument(
        "--door-open-durations",
        type=float,
        nargs="+",
        help="Run one or more cases with these door open durations in seconds, e.g. 30 60 120.",
    )
    parser.add_argument("--parallel", action="store_true", help="Run multiple requested cases in parallel processes.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes for --parallel.")
    parser.add_argument(
        "--run-version",
        help="Append a stable version tag to output files, e.g. 2 creates *_2.csv and rerunning 2 overwrites it.",
    )
    args = parser.parse_args()

    door_durations = _door_durations_from_args(args)
    if door_durations is None:
        cfg = load_config(args.config)
        apply_output_version(cfg, args.run_version)
        history = run_simulation(cfg)
        save_plot(history, cfg["output"]["plot_file"])
        save_csv(history, cfg["output"]["csv_file"])
        _print_final_state(history[-1], cfg["output"]["plot_file"], cfg["output"]["csv_file"])
        return

    if len(door_durations) == 1:
        result = run_case_from_config_path(args.config, door_durations[0], args.run_version)
        _print_case_result(result)
        return

    workers = 1
    if args.parallel:
        workers = args.workers or min(len(door_durations), os.cpu_count() or 1)
    elif args.workers is not None and args.workers > 1:
        workers = args.workers

    print(f"Running {len(door_durations)} door-duration cases with workers={workers}")
    results = run_cases(args.config, list(door_durations), workers=workers, run_version=args.run_version)
    for result in results:
        _print_case_result(result)


if __name__ == "__main__":
    main()
