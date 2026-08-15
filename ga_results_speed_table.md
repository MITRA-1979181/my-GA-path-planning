# GA Planner — speed / generations sweep (2026-08-16)
# Autoware planning_simulator, sample_vehicle, KNU IT_map (405 m)
# Metric: mean |lateral error| vs lane_centerline.csv

| V_NOMINAL | GEN | straight 10-140 | straight 255-400 | curves 150-250 | compute | full route |
|-----------|-----|-----------------|------------------|----------------|---------|------------|
| 1.0 | 15 |  4.7 cm |   4.0 cm | 31.9 cm | 200 ms | yes |
| 2.0 | 15 |  4.2 cm |   4.5 cm | 33.1 cm | 200 ms | yes |
| 3.0 | 15 |  4.0 cm |   5.3 cm | 37.8 cm | 200 ms | yes |
| 4.0 |  8 | 28.8 cm |  90.4 cm | 63.3 cm | 100 ms | NO (m313) |
| 4.0 | 15 | 10.6 cm |  19.7 cm | 45.1 cm | 200 ms | yes |
| 4.0 | 30 |  7.4 cm |  20.8 cm | 40.1 cm | 393 ms | yes |
| 6.0 | 15 | 76.3 cm | 144.8 cm | 74.3 cm | 200 ms | NO (m286) |

## Findings
- Solution quality (generations) matters more than cycle time:
  30 generations at 393 ms beat 15 at 200 ms despite 2x path staleness.
- Operating setting: V_NOMINAL 3.0 m/s (11 km/h), 15 generations.
- Physical limit of this route: tightest curve R=8.3 m -> 15 km/h at a_lat=2.0.

## After elite-seeding and goal-stop fixes (2026-08-16, v24/v25)
| V_NOMINAL | GEN | straight 10-140 | straight 255-400 | curves 150-250 | overall | full route |
|-----------|-----|-----------------|------------------|----------------|---------|------------|
| 3.0 | 15 |  3.5 cm |  6.8 cm | 30.6 cm | 13.5 cm | yes, stops at goal |
| 4.0 | 15 | 18.1 cm | 59.2 cm | 47.9 cm | 39.6 cm | oscillates, overshoots goal |

Fixes applied in this session:
1. compute_target_speed() was defined but never called -> curvature-adaptive
   speed profile was completely inactive. Now wired into run_ga.
2. Elite-seeding guard required 120 reference points ahead, so past meter 372
   no ref-elites were built and the population was 100% random
   (best_fitness dropped 0.94 -> 0.62, blue path became a curve).
   Guard relaxed to snap_idx + 5.
3. Goal-stop condition required BOTH reference index AND euclidean distance
   < 2.0 m; with a mid-route goal it never fired. Now index-only.

Attempts that did NOT raise the stability ceiling above 3.0 m/s:
- GENERATIONS 8 / 15 / 30
- mpc_weight_steer_rate 0.0 -> 5.0
- input_delay 0.10 / 0.24 / 0.40
- curvature_smoothing_num_ref_steer 15 -> 5
- A_LAT_MAX 2.0 -> 1.0

Operating setting: V_NOMINAL = 3.0 m/s (11 km/h), GENERATIONS = 15.
Route physical limit: tightest curve R = 8.3 m -> 15 km/h at a_lat = 2.0 m/s^2.
Open issue: ~50 cm one-sided bias in the curve at meter 210-230, resistant to
eight separate hypotheses; GA path verified correct by offline reconstruction.
