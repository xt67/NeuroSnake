# Local verification — 18 November 2026

## NeuroSnake

| Run | Training games | Evaluation games | Mean | Standard deviation | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy, seed 42, 12×12 board | 100 | 100 | 19.46 | 6.84 | 45 |
| NumPy, seed 42, 12×12 board | 500 | 100 | 23.24 | 7.62 | 42 |

Evaluation used seeds 10000–10099 and greedy actions without exploration. Runs
have different exploration schedules because epsilon decay depends on total
training games. These are two runs, not checkpoint comparisons from one run.
No further hyperparameter selection was performed after these evaluations.

The 500-game run's files are in `artifacts/numpy-500/`: inference weights,
configuration, score CSV, learning curve PNG and evaluation JSON.

Five NeuroSnake unit tests passed. NumPy gradients were checked against
finite differences for every parameter in a small network. The game tests covered
food, walls, the moving tail, full-board completion and starvation. Model loading
was checked against saved predictions. Pygame rendered two frames using SDL's
headless driver; interactive keyboard play has not been manually verified.
The PyTorch backend completed a two-game training and checkpoint smoke run.

Suggested resume statement based on the measured 500-game run:

> Implemented a deep Q-learning Snake agent with a NumPy neural network and manual
> backpropagation, averaging 23.24 food pickups across 100 evaluation games after
> 500 training games on a 12×12 board.

