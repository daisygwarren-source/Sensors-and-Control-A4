## cube (20 randomised trials per row, same seeds in every row)

| condition | picked | placed | grasp offset mm | cube est err mm | cube NEES (3) | NIS rejects | q err urad | qd err mrad/s | joint NEES (2) | track rms mrad | track max mrad | track max grasp/lift mrad | torque rms % | torque rough % | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CV model (original), nominal | 100% | 100% | 1.6 / 3.1 | 6.44 / 12.8 | 7.5 | 0.0 | 5.6 | 2.63 | 1.65 | 0.299 | 0.14 | 2.48 | 8.3 | 0.706 | 17.3 |
| CV model (original), nudged | 90% | 90% | 1.7 / 5.6 | 6.72 / 13.9 | 33.3 | 3.0 | 6.0 | 3.35 | 3.15 | 44.889 | 115.97 | 2.24 | 13.4 | 0.716 | 17.4 |
| CP model, R x 0.1 | 100% | 100% | 0.3 / 0.5 | 0.81 / 0.9 | 81.5 | 0.0 | 5.5 | 2.53 | 1.59 | 0.266 | 0.14 | 2.09 | 8.3 | 0.692 | 17.0 |
| CP model, R x 1 | 100% | 100% | 0.3 / 0.5 | 0.69 / 0.9 | 9.5 | 0.0 | 5.5 | 2.52 | 1.58 | 0.270 | 0.14 | 2.11 | 8.3 | 0.690 | 17.0 |
| CP model, R x 4 (default) | 100% | 100% | 0.3 / 0.4 | 0.68 / 0.9 | 2.9 | 0.0 | 5.5 | 2.51 | 1.58 | 0.276 | 0.14 | 2.13 | 8.3 | 0.688 | 17.0 |
| CP model, R x 10 | 100% | 100% | 0.3 / 0.5 | 0.64 / 0.9 | 1.5 | 0.0 | 5.5 | 2.51 | 1.58 | 0.278 | 0.14 | 2.14 | 8.3 | 0.685 | 17.0 |
| CP model, R x 4, nudged | 100% | 100% | 0.3 / 0.4 | 0.66 / 0.9 | 87.7 | 3.0 | 5.5 | 2.51 | 1.58 | 0.276 | 0.14 | 2.13 | 8.3 | 0.689 | 17.0 |
