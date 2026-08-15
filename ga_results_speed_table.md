# GA Planner — speed / generations sweep (2026-08-16)
# Simulator: Autoware planning_simulator, sample_vehicle, KNU IT_map (405 m)
# Metric: mean |lateral error| vs lane_centerline.csv

| V_NOMINAL | GENERATIONS | straight 10-140 | straight 255-400 | curves 150-250 | compute | full route |
|-----------|-------------|-----------------|------------------|----------------|---------|------------|
| 1.0 | 15 |  4.7 cm |  4.0 cm | 31.9 cm | 200 ms | yes |
| 2.0 | 15 |  4.2 cm |  4.5 cm | 33.1 cm | 200 ms | yes |
| 3.0 | 15 |  4.0 cm |  5.3 cm | 37.8 cm | 200 ms | yes |
| 4.0 |  8 | 28.8 cm | 90.4 cm | 63.3 cm | 100 ms | NO (m313) |
| 4.0 | 15 | 10.6 cm | 19.7 cm | 45.1 cm | 200 ms | yes |
| 4.0 | 30 |  7.4 cm | 20.8 cm | 40.1 cm | 393 ms | yes |
| 6.0 | 15 | 76.3 cm | 144.8 cm | 74.3 cm | 200 ms | NO (m286) |

## Findings
- Solution quality (generations) matters more than cycle time.
  30 generations at 393 ms beat 15 at 200 ms, despite 2x path staleness.
- Stability ceiling ~3.0 m/s (11 km/h) with 15 generations.
- Physical limit of this route: tightest curve R=8.3 m -> 15 km/h at a_lat=2.0.
- MPC insensitive to input_delay across 0.10-0.40 s (no measurable change).
- Curve bias (~60 cm) survived 8 hypotheses; GA path verified correct offline.
