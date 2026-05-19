from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .simulation import run_simulation, save_csv, save_plot


@dataclass(frozen=True)
class CaseResult:
    name: str
    door_open_duration_s: float | None
    plot_file: str
    csv_file: str
    final_room_c: float
    final_dock_c: float
    final_sink_c: float
    final_cop_system: float
    final_m_air_kg_s: float
    final_m_ref_kg_s: float


def _format_duration_tag(duration_s: float) -> str:
    if float(duration_s).is_integer():
        return f"{int(duration_s)}s"
    return f"{duration_s:g}s".replace(".", "p")


def _with_suffix(path: str | Path, suffix: str) -> str:
    out_path = Path(path)
    return str(out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix}"))


def _format_version_tag(run_version: str | int | None) -> str | None:
    if run_version is None:
        return None
    tag = str(run_version).strip()
    if not tag:
        return None
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in tag)


def apply_output_version(config: dict[str, Any], run_version: str | int | None) -> None:
    version_tag = _format_version_tag(run_version)
    if version_tag is None:
        return

    output_cfg = config.setdefault("output", {})
    output_cfg["plot_file"] = _with_suffix(output_cfg["plot_file"], version_tag)
    output_cfg["csv_file"] = _with_suffix(output_cfg["csv_file"], version_tag)


def apply_door_open_duration(config: dict[str, Any], duration_s: float) -> None:
    infiltration_cfg = config.setdefault("disturbances", {}).setdefault("infiltration", {})
    duration = max(float(duration_s), 0.0)

    schedule = infiltration_cfg.get("schedule")
    events = None
    if isinstance(schedule, dict):
        events = schedule.get("events")
    if events is None:
        events = infiltration_cfg.get("events")

    if events:
        for event in events:
            t_open = float(event.get("t_open_s", infiltration_cfg.get("t_open_s", 0.0)))
            event["t_open_s"] = t_open
            event["t_close_s"] = t_open + duration
        return

    t_open = float(infiltration_cfg.get("t_open_s", infiltration_cfg.get("start_time_s", 0.0)))
    infiltration_cfg["t_open_s"] = t_open
    infiltration_cfg["t_close_s"] = t_open + duration
    infiltration_cfg["open_duration_s"] = duration


def build_case_config(
    base_config: dict[str, Any],
    door_open_duration_s: float | None,
    run_version: str | int | None = None,
) -> tuple[str, dict[str, Any]]:
    config = deepcopy(base_config)
    if door_open_duration_s is None:
        apply_output_version(config, run_version)
        return "base", config

    tag = f"door_{_format_duration_tag(door_open_duration_s)}"
    apply_door_open_duration(config, door_open_duration_s)
    output_cfg = config.setdefault("output", {})
    output_cfg["plot_file"] = _with_suffix(output_cfg["plot_file"], tag)
    output_cfg["csv_file"] = _with_suffix(output_cfg["csv_file"], tag)
    apply_output_version(config, run_version)
    return tag, config


def run_case(config: dict[str, Any], name: str, door_open_duration_s: float | None = None) -> CaseResult:
    history = run_simulation(config)
    save_plot(history, config["output"]["plot_file"])
    save_csv(history, config["output"]["csv_file"])

    last = history[-1]
    return CaseResult(
        name=name,
        door_open_duration_s=door_open_duration_s,
        plot_file=str(Path(config["output"]["plot_file"]).resolve()),
        csv_file=str(Path(config["output"]["csv_file"]).resolve()),
        final_room_c=float(last["room_c"]),
        final_dock_c=float(last["dock_c"]),
        final_sink_c=float(last["sink_c"]),
        final_cop_system=float(last["cop_system"]),
        final_m_air_kg_s=float(last["m_air_kg_s"]),
        final_m_ref_kg_s=float(last["m_ref_kg_s"]),
    )


def run_case_from_config_path(
    config_path: str | Path,
    door_open_duration_s: float | None = None,
    run_version: str | int | None = None,
) -> CaseResult:
    base_config = load_config(config_path)
    name, config = build_case_config(base_config, door_open_duration_s, run_version)
    return run_case(config, name, door_open_duration_s)


def run_cases(
    config_path: str | Path,
    door_open_durations_s: list[float | None],
    *,
    workers: int = 1,
    run_version: str | int | None = None,
) -> list[CaseResult]:
    if workers <= 1 or len(door_open_durations_s) <= 1:
        return [run_case_from_config_path(config_path, duration, run_version) for duration in door_open_durations_s]

    results: list[CaseResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_case_from_config_path, config_path, duration, run_version): duration
            for duration in door_open_durations_s
        }
        for future in as_completed(futures):
            results.append(future.result())

    order = {duration: idx for idx, duration in enumerate(door_open_durations_s)}
    return sorted(results, key=lambda result: order[result.door_open_duration_s])
