## cube (8 randomised trials per row, same seeds in every row)

| condition | picked | placed | grasp offset mm | cube est err mm | cube NEES (3) | NIS rejects | q err urad | qd err mrad/s | joint NEES (2) | track rms mrad | track max mrad | track max grasp/lift mrad | torque rms % | torque rough % | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CV model (original), nominal | 100% | 100% | 1.1 / 2.6 | 4.68 / 9.6 | 8.5 | 0.0 | 5.6 | 2.63 | 1.65 | 0.356 | 0.14 | 3.08 | 8.3 | 0.706 | 17.3 |
| CV model (original), nudged | 88% | 88% | 1.7 / 5.6 | 6.22 / 13.9 | 32.6 | 3.0 | 5.6 | 2.82 | 1.75 | 73.822 | 146.89 | 2.55 | 15.3 | 0.673 | 17.5 |
| CP model, R x 0.1 | 100% | 100% | 0.3 / 0.4 | 0.76 / 0.9 | 92.3 | 0.0 | 5.5 | 2.52 | 1.58 | 0.263 | 0.14 | 2.02 | 8.2 | 0.685 | 17.1 |
| CP model, R x 1 | 100% | 100% | 0.3 / 0.4 | 0.70 / 0.9 | 10.8 | 0.0 | 5.6 | 2.57 | 1.61 | 0.247 | 0.14 | 2.05 | 8.2 | 0.684 | 17.1 |
| CP model, R x 4 (default) | 100% | 100% | 0.3 / 0.4 | 0.64 / 0.9 | 3.3 | 0.0 | 5.5 | 2.51 | 1.58 | 0.270 | 0.14 | 2.07 | 8.2 | 0.681 | 17.1 |
| CP model, R x 10 | 100% | 100% | 0.3 / 0.4 | 0.58 / 0.9 | 1.7 | 0.0 | 5.5 | 2.51 | 1.58 | 0.272 | 0.14 | 2.10 | 8.2 | 0.683 | 17.1 |
| CP model, R x 4, nudged | 100% | 100% | 0.3 / 0.4 | 0.78 / 0.9 | 87.2 | 3.0 | 5.5 | 2.50 | 1.57 | 0.275 | 0.14 | 2.10 | 8.2 | 0.679 | 17.1 |
