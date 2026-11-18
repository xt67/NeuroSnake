# NeuroSnake

A playable Snake game and Double DQN agent with two interchangeable neural networks:
PyTorch for a familiar baseline, and NumPy with manually implemented forward propagation,
Huber loss, backpropagation, gradient clipping and Adam updates.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m neurosnake play
```

Arrow keys move; Space pauses; R restarts; Escape exits.

## Train and watch

```powershell
# Optional first baseline (requires pip install torch)
python -m neurosnake train --backend torch --games 500 --output artifacts/torch
python -m neurosnake play --backend torch --model artifacts/torch/model.pt

# NumPy-only neural network
python -m neurosnake train --games 500 --output artifacts/numpy
python -m neurosnake play --model artifacts/numpy/model.npz
python -m neurosnake evaluate --model artifacts/numpy/model.npz --games 100 --output artifacts/numpy/evaluation.json
python -m unittest discover -s tests -v
```

Training is headless. Each run saves a final inference model, configuration, per-game
scores and a learning curve. Checkpoints contain inference weights, not resumable
optimizer/replay state. Use the same `--size` for training and evaluation.

## How it learns

The 11 binary inputs are danger straight/right/left; direction left/right/up/down;
and food left/right/up/down. The network is 11 → 128 ReLU → 3 Q values, representing
straight/right/left. Rewards are +10 for food, -10 for a collision or starvation,
and 0 otherwise. Starvation ends a game after 100 × snake length steps without food.
A full board ends successfully. The tail may be entered when it moves away.

An epsilon-greedy policy explores initially. Experience replay stores 50,000
transitions; batches of 64 train every four moves. A frozen target network updates
every 500 moves. Double DQN targets use discount 0.95. All randomness is seeded.
Evaluation disables exploration and defaults to new seeds starting at 10000.

## Measured local result

The NumPy agent trained for 500 games (seed 42, 12×12 board), then averaged **23.24**
food pickups over 100 evaluation games (seeds 10000–10099, no exploration).
Standard deviation: 7.62; best evaluation score: 42. This is one training seed,
not a multi-seed benchmark. See `RESULTS.md` for verification details.

The trained local model can be watched with:

```powershell
python -m neurosnake play --model artifacts/numpy-500/model.npz
```

Generated models and datasets are ignored by Git. A fresh clone needs training first.
Requirements select NumPy 1.26 for compatibility with older PyTorch binaries. The
local NumPy backend also passed with NumPy 2.5.4; the installed PyTorch 2.2 emits a
NumPy ABI warning. Use the isolated environment above to avoid that combination.

Read `neurosnake/network.py` for every NumPy gradient. Tests compare those gradients
against numerical finite differences and check the rules and checkpoint round trip.
Learning speed and scores depend on seed, board size and training budget. Report
evaluation scores, not the highest training score; do not fill resume metrics before
running evaluation. The whole-board input experiment remains a possible extension.

## References

Original implementation inspired by the task, not copied from these resources:
- https://github.com/patrickloeber/snake-ai-pytorch
- https://github.com/karpathy/micrograd
- https://arxiv.org/abs/1312.5602

Companion project: [Heartbeat Classifier](https://github.com/xt67/Heartbeat-Classifier).
