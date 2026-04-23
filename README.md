# Cascade Refrigeration Dynamic Simulator

This project simulates the dynamic performance of a cascade refrigeration system made of:

- a bottom reverse Brayton air cycle
- a top vapor-compression ammonia cycle
- an explicit loading-dock evaporator branch on the ammonia side

The implementation uses:

- `CoolProp` for thermophysical properties
- `numpy` for array math
- `matplotlib` for plots
- a custom Newton-Raphson nonlinear solver with finite-difference Jacobian

## Model scope

The model is organized as a semi-explicit DAE solved with backward Euler in time:

- Dynamic states:
  - condenser sink outlet / lumped sink temperature
- Algebraic variables:
  - refrigerated room / compressor inlet temperature from the room heater load
  - air-cycle regenerator outlet temperatures
  - air temperature entering the turbine
  - air supply temperature to the cold room
  - ammonia evaporating temperature
  - ammonia mass flow through the expansion valve

Component models:

- compressor / turbine: isentropic efficiency
- regenerator / cascade HX / condenser: `UA-LMTD`
- room load: heater model, `Q_RS = m_air * (h_room_return - h_supply)`
- loading dock branch: direct heater/load on the refrigerant side, `Q_LD(t)`
- expansion valve: isenthalpic with linear pressure-drop flow equation
- bottom-side air flow: compressor head-to-volumetric-flow polynomial with
  suction-density conversion to mass flow, driven by compressor head only
- ammonia compressor discharge pressure: fixed compressor pressure ratio when
  `vcc_cycle.compressor_pressure_ratio` is provided

The bottom-side air-cycle layout used by the model is:

```text
6 -> compressor -> 2 -> cascade HX -> 3 -> regenerator hot side -> 4
4 -> turbine -> 5 -> refrigerated room heater/load -> room_c
room_c -> regenerator cold side -> 6

NH3 evaporator load:

```text
Q_evap_total = Q_cascade + Q_LD
```
```

So the cascade HX cools the compressor discharge before the regenerator, and
the regenerator LMTD is based on:

```text
LMTD_reg = LMTD(T3 - T6, T4 - Troom)
```

The bottom-side air mass flow is computed from the compressor isentropic head:

```text
h_is = h(P2, s1) - h1
H = h_is / g
Q = a*H^3 + b*H^2 + c*H + d
m_air = rho_suction * Q
```

The active compressor map expects:

```text
H in meters of isentropic head
Q in m^3/s
```

The room controller acts on `air_cycle.pressure_ratio`, so compressor head changes the calculated bottom-cycle mass flow.

For the ammonia compressor, the configured reference case now uses:

```text
p_cond = compressor_pressure_ratio * p_evap
tcond = Tsat(p_cond)
h8 = h7 + (h8s - h7) / compressor_eta_is
```

with `compressor_pressure_ratio = 4.387` and `compressor_eta_is = 0.85`.
This uses the supplied compressor pressure ratio directly for the NH3 discharge
pressure instead of deriving discharge pressure only from condenser saturation
temperature.

## Run

```powershell
python -m src.cascade_dynamics.main --config config/paper_reference_case.json
```

The script prints a short summary and writes plots into `outputs/`.

Before the transient run starts, `simulation.startup_initialization` can run a
separate startup design-point solve. This mode does not advance time. It fixes
the configured targets, solves selected startup-only free variables, writes those
values back into the in-memory config, then freezes them for the dynamic time
loop. If `freeze_solved_parameters` is true, controllers whose actuator path was
solved during startup report the frozen value instead of overwriting it.

The startup problem is configured by:

- `targets`: design-point outputs such as room temperature, sink temperature,
  evaporator temperature, condenser temperature, refrigerant mass flow, and air
  mass flow
- `free_state_variables`: internal algebraic temperatures that may move while
  fitting the operating point
- `free_parameters`: config paths that may be solved during startup, such as
  condenser UA, cascade HX UA, expansion-valve opening, VCC compressor pressure
  ratio, air-cycle pressure ratio, or air-compressor map speed fraction

Startup solved parameters are also copied into the first CSV row as
`startup_solved_*` snapshot columns. Disable startup initialization with:

```json
"startup_initialization": {
  "enabled": false
}
```

Temperature inputs and outputs in the JSON config, CSV, plots, and printed
summary use degrees Celsius. The thermodynamic property calls convert Celsius to
Kelvin internally only where CoolProp requires absolute temperature.

## Disturbance Input

Infiltration is configured in JSON under `disturbances.infiltration`.

```json
{
  "enabled": true,
  "start_time_s": 900.0,
  "delay_s": 120.0,
  "ramp_time_s": 30.0,
  "hold_time_s": 60.0,
  "magnitude_w": 10000.0,
  "room_fraction": 1.0,
  "dock_fraction": -1.0
}
```

The disturbance starts only after `start_time_s + delay_s`, ramps up linearly over
`ramp_time_s`, stays at full magnitude for `hold_time_s`, then ramps back down to
zero over the same `ramp_time_s`. With the fractions above, it increases room load
and decreases dock load by the same amount.

## PI Control

PI controllers are configured in JSON under `control.controllers`.
Each controller reads one output from the previous time step and writes one actuator path before the next Newton solve.

The implemented discrete form is:

```text
error = measurement - setpoint
if action == "reverse": error = -error
integral = integral + error * dt
output = bias + gain * error + gain * integral / Ti
output = clamp(output, u_min, u_max)
```

The current controller mapping follows the diagram at the level exposed by this lumped model:

- `B1_room_to_air_compressor`: room temperature to air-cycle mass flow, direct action
- `B3_sink_to_condenser`: sink/condenser temperature to condenser UA, direct action
- `VC5_dock_to_dock_evaporator`: loading-dock temperature to dock evaporator UA, direct action
- `VC4_expansion_valve`: evaporating temperature to superheat, reverse action
- `B4_cascade_hx_outlet`: cascade heat-exchanger outlet temperature to cascade HX UA, reverse action

The last two are configured as reverse action to match the requested exceptions.

## Notes

- The paper title used as design context is:
  `Cascade refrigeration system with inverse Brayton cycle on the cold side`
- `config/paper_reference_case.json` uses values recovered from Table 2 of the paper, then translates them into this simplified transient model.
- Because the paper is primarily a steady-state design study and includes humid-air expansion details and a loading-dock evaporator branch that are not yet fully represented here, a few values are derived approximations:
  - Brayton pressure ratio from the paper temperatures and turbine efficiency under a dry-air fit
  - ammonia compressor pressure ratio and isentropic efficiency supplied as `4.387` and `0.85`
  - UA values back-calculated from reference duties and terminal temperatures
- The code is intentionally modular so you can later add more detailed control volumes, pressure dynamics, frost, or more detailed heat exchanger discretization.
