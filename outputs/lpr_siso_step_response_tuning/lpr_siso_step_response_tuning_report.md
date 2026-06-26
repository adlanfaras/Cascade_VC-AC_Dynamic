# LPR SISO Step-Response Tuning Report

## Method

1. Each `*_lpr.json` case was treated as a SISO loop: `room_c` controls `air_cycle.compressor_mass_flow.speed_rpm`.
2. Door infiltration and load changes were disabled for identification so the measured response came only from an air-compressor speed step.
3. A startup solve established the nominal air-compressor speed. The transient identification run then stepped that speed upward and recorded `room_c`.
4. The open-loop Ziegler-Nichols reaction-curve method was applied directly to the step response:

   - draw the tangent at the steepest point of the process reaction curve
   - `L` is the time from the input step to the tangent intersection with the initial temperature level
   - `T` is the time between the tangent intersections with the initial and final temperature levels
   - `K = delta T_room / delta N_air_compressor`

5. The PI controller was tuned with the open-loop Ziegler-Nichols PI rule:

   `Kc = 0.9 * T / (abs(K) * L)`

   `Ti = 3.33 * L`

The process gain is negative because increasing compressor speed cools the room. The existing controller uses `action: direct`, so the applied controller gain is positive.

## Results

| Config | Step [rpm] | K [C/rpm] | T [s] | L [s] | Tangent slope [C/s] | gain [rpm/K] | Ti [min] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LD_Infiltration_Qvc100_Qld25_1_lpr.json | 1000.0 | -0.00370007 | 246.6 | 30.0 | -0.0150044 | 1999.41 | 1.67 |
| RS_Infiltration_Qvc100_Qld15_2_lpr.json | 1000.0 | -0.00416414 | 179.1 | 30.0 | -0.0232472 | 1290.48 | 1.67 |
| RS_Infiltration_Qvc100_Qld15_4_lpr.json | 1000.0 | -0.00416414 | 179.1 | 30.0 | -0.0232472 | 1290.48 | 1.67 |
| RS_Infiltration_Qvc100_Qld15_8_lpr.json | 1000.0 | -0.00416414 | 179.1 | 30.0 | -0.0232472 | 1290.48 | 1.67 |
| RS_Infiltration_Qvc100_Qld20_2_lpr.json | 1000.0 | -0.00527985 | 267.9 | 30.0 | -0.0197081 | 1522.21 | 1.67 |
| RS_Infiltration_Qvc100_Qld20_4_lpr.json | 1000.0 | -0.00527985 | 267.9 | 30.0 | -0.0197081 | 1522.21 | 1.67 |
| RS_Infiltration_Qvc100_Qld20_8_lpr.json | 1000.0 | -0.00527985 | 267.9 | 30.0 | -0.0197081 | 1522.21 | 1.67 |
| RS_Infiltration_Qvc100_Qld25_2_lpr.json | 1000.0 | -0.00370007 | 246.6 | 30.0 | -0.0150044 | 1999.41 | 1.67 |
| RS_Infiltration_Qvc100_Qld25_4_lpr.json | 1000.0 | -0.00370007 | 246.6 | 30.0 | -0.0150044 | 1999.41 | 1.67 |
| RS_Infiltration_Qvc100_Qld25_8_lpr.json | 1000.0 | -0.00370007 | 246.6 | 30.0 | -0.0150044 | 1999.41 | 1.67 |

## Outputs

- Identification plots: `outputs\lpr_siso_step_response_tuning` with suffix `_identification.png`.
- Publication-style Ziegler-Nichols reaction-curve panels: `outputs\lpr_siso_step_response_tuning` with suffix `_zn_reaction_curve.png`.
- Smoother representative step-response panel generated with `dt = 15 s`: `outputs\lpr_siso_step_response_tuning\RS_Infiltration_Qvc100_Qld25_4_lpr_zn_reaction_curve_smooth_dt15.png`.
- Representative PI control performance plot generated with `dt = 2.5 s` for a +1% room-load step: `outputs\lpr_siso_step_response_tuning\RS_Infiltration_Qvc100_Qld25_4_lpr_pi_room_load_step_p1pct.png`.
- Numeric summary: `outputs\lpr_siso_step_response_tuning\lpr_siso_step_response_tuning_summary.csv`.
- Closed-loop verification was skipped for the full matrix to avoid a long nonlinear simulation batch; the tuning evidence is the open-loop Ziegler-Nichols reaction curve.
