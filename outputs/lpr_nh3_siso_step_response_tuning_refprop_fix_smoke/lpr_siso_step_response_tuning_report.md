# LPR SISO NH3 Compressor Step-Response Tuning Report

## Method

Property backend: `refprop`.
REFPROP path: `C:\Program Files (x86)\REFPROP`.

1. Each `*_lpr.json` case was treated as a SISO loop: `room_c` controls `vcc_cycle.compressor.speed_rpm`.
2. Door infiltration and load changes were disabled for identification so the measured response came only from a NH3 compressor speed step.
3. A startup solve established the nominal NH3 compressor speed. The transient identification run then stepped that speed upward and recorded `room_c`.
4. The open-loop Ziegler-Nichols reaction-curve method was applied directly to the step response:

   - draw the tangent at the steepest point of the process reaction curve
   - `L` is the time from the input step to the tangent intersection with the initial temperature level
   - `T` is the time between the tangent intersections with the initial and final temperature levels
   - `K = delta T_room / delta N_NH3`

5. The PI controller was tuned with the open-loop Ziegler-Nichols PI rule:

   `Kc = 0.9 * T / (abs(K) * L)`

   `Ti = 3.33 * L`

The selected room controller uses `action: direct` for this actuator profile; the applied controller gain is reported as a positive magnitude.

## Results

| Config | Step [rpm] | K [C/rpm] | T [s] | L [s] | Tangent slope [C/s] | gain [rpm/K] | Ti [min] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RS_Infiltration_Qvc100_Qld25_4_lpr.json | 100.0 | -0.0116503 | 486.6 | 20.1 | -0.00239408 | 1870.60 | 1.12 |

## Outputs

- Identification plots: `outputs\lpr_nh3_siso_step_response_tuning_refprop_fix_smoke` with suffix `_identification.png`.
- Publication-style Ziegler-Nichols reaction-curve panels: `outputs\lpr_nh3_siso_step_response_tuning_refprop_fix_smoke` with suffix `_zn_reaction_curve.png`.
- Numeric summary: `outputs\lpr_nh3_siso_step_response_tuning_refprop_fix_smoke\lpr_siso_step_response_tuning_summary.csv`.
- Closed-loop verification was skipped for the full matrix to avoid a long nonlinear simulation batch; the tuning evidence is the open-loop Ziegler-Nichols reaction curve.
