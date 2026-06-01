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

Use one of the included reference configs:

```text
config/paper_reference_case.json
config/paper_reference_case_dock_infiltration.json
```

The first config models room-to-dock door infiltration. The second config models loading-dock-only infiltration from ambient through the dock door.

## Run One Case

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case.json
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json
```

## Run Door-Duration Cases

Run one modified door-open duration:

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case.json --door-open-duration 30
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-duration 30
```

Run 30 s, 60 s, and 120 s cases:

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case.json --door-open-durations 30 60 120
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-durations 30 60 120
```

## Parallel Runs

Run multiple door-duration cases in parallel:

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case.json --door-open-durations 30 60 120 --parallel --workers 3
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-durations 30 60 120 --parallel --workers 3
```

## Versioned Outputs

Add a stable version tag to output filenames:

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case.json --run-version 2
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --run-version 2
```

Combine versioned outputs with parallel door-duration runs:

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case.json --door-open-durations 30 60 120 --parallel --workers 3 --run-version 2
```

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case_dock_infiltration.json --door-open-durations 30 60 120 --parallel --workers 3 --run-version 2
```

With `--run-version 2`, outputs are written with suffixes such as:

```text
outputs/paper_reference_case_door_30s_2.csv
outputs/paper_reference_case_door_30s_2.png
```

Rerunning the same version overwrites that version's files.
