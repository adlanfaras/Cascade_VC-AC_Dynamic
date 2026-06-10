# Cascade Refrigeration Dynamic Simulator

Dynamic simulator for a cascade refrigeration system with:

- reverse-Brayton air cycle on the cold side
- ammonia vapor-compression cycle on the warm side
- loading-dock evaporator branch
- optional door-opening infiltration disturbance cases

The simulator writes CSV results and plots to `outputs/`.

## Setup

```powershell
pip install -r requirements.txt
```

## Configs

Use one of the included refrigerated-space or loading-dock infiltration configs:

```text
config/RS_Infiltration_Qvc100_Qld25_2.json
config/RS_Infiltration_Qvc100_Qld25_4.json
config/RS_Infiltration_Qvc100_Qld25_8.json
config/LD_Infiltration_Qvc100_Qld25_2.json
config/LD_Infiltration_Qvc100_Qld25_4.json
config/LD_Infiltration_Qvc100_Qld25_8.json
```

The full config matrix covers matching Qvc/Qld load variations for refrigerated-space (`RS_...`) and loading-dock (`LD_...`) infiltration. Loading-dock configs apply ambient infiltration to the dock load only. All included infiltration configs use an effectiveness of `0.96`, a `2.8 m x 3.3 m` door, and a one-opening-per-hour schedule.

Door-opening disturbances can be configured with explicit open/close times, or with a repeating start/duration/interval schedule. Durations may be provided in seconds or hours:

```json
"infiltration": {
  "enabled": true,
  "model": "tian_unsteady",
  "schedule": {
    "start_time_s": 5.0,
    "ramp_time_s": 2.0,
    "open_duration_s": 12.0,
    "interval_s": 3600.0,
    "repeat_count": 1,
    "opening_fraction": 1.0
  },
  "effectiveness": 0.96,
  "door": {
    "width_m": 2.8,
    "height_m": 3.3
  }
}
```

Here `interval_s` is the time from one opening start to the next opening start. Use `door_ramp_open_time_s` and `door_ramp_close_time_s` if the opening and closing ramps should have different durations. Existing `t_open_s`/`t_close_s` configs remain supported.

## Run One Case

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json
```

```powershell
python -m src.cascade_dynamics.main --config config/LD_Infiltration_Qvc100_Qld25_8.json
```

## Run Door-Duration Cases

Run one modified door-open duration:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-duration 30
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-duration 30
```

Run 30 s, 60 s, and 120 s cases:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-durations 30 60 120
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-durations 30 60 120
```

## Parallel Runs

Run multiple door-duration cases in parallel:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-durations 30 60 120 --parallel --workers 3
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-durations 30 60 120 --parallel --workers 3
```

With only three door-duration cases, the outer batch can only keep three processes busy. You can also experiment with parallel finite-difference Jacobian work inside each simulation:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-durations 30 60 120 --parallel --workers 3 --jacobian-workers 3
```

Benchmark `--jacobian-workers 2`, `3`, or `4` before using it for production runs. Very high values can be slower because each timestep solves a small 10-variable system.

## Versioned Outputs

Add a stable version tag to output filenames:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --run-version 2
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --run-version 2
```

Combine versioned outputs with parallel door-duration runs:

```powershell
python -m src.cascade_dynamics.main --config config/RS_Infiltration_Qvc100_Qld25_8.json --door-open-durations 30 60 120 --parallel --workers 3 --run-version 2
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-durations 30 60 120 --parallel --workers 3 --run-version 2
```

With `--run-version 2`, outputs are written with suffixes such as:

```text
outputs/RS_Infiltration_Qvc100_Qld25_8_door_30s_2.csv
outputs/RS_Infiltration_Qvc100_Qld25_8_door_30s_2.png
```

Rerunning the same version overwrites that version's files.
