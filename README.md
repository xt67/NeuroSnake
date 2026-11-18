# NeuroSnake

NeuroSnake is a reinforcement learning project where an AI agent learns to play the classic Snake game using **Double Deep Q-Learning (Double DQN)**.

The project features two neural network implementations: PyTorch and a NumPy-based network with manually implemented forward propagation and backpropagation.

## Features

- Playable Snake game built with Pygame.
- AI agent trained using Double Deep Q-Learning.
- PyTorch and NumPy neural network implementations.
- Manual backpropagation in the NumPy implementation.
- Experience replay and target network.
- Automated training and model evaluation.
- Training progress visualisation.
- Browser-based demonstration of the trained AI.

## Tech Stack

- Python
- NumPy
- PyTorch (optional)
- Pygame
- Matplotlib
- JavaScript

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/xt67/NeuroSnake.git
cd NeuroSnake
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Play Snake**

```bash
python -m neurosnake play
```

**5. Train the AI**

Using NumPy:

```bash
python -m neurosnake train --backend numpy --games 500
```

Using PyTorch (install PyTorch separately):

```bash
python -m neurosnake train --backend torch --games 500 --output artifacts/torch
```

**6. Watch the trained AI**

```bash
python -m neurosnake play --model artifacts/numpy/model.npz
```

**7. Evaluate the trained model**

```bash
python -m neurosnake evaluate --model artifacts/numpy/model.npz --games 100
```

## Project Structure

```text
NeuroSnake/
├── neurosnake/  # Game environment and neural networks
├── tests/       # Unit tests
├── web/         # Browser-based demonstration
├── RESULTS.md   # Training results and verification
├── requirements.txt
└── README.md
```

## Training Results

The NumPy agent was trained for 500 games on a 12 × 12 board.

| Metric | Result |
|---|---:|
| Evaluation games | 100 |
| Average score | 23.24 |
| Best score | 42 |

These results are from a single training run. See [RESULTS.md](RESULTS.md) for additional details.

## Current Status

The game environment, Double DQN training pipeline, NumPy and PyTorch implementations, and browser demonstration are implemented.

The trained NumPy model can be used to play and evaluate the game.

---

Developed by [xt67](https://github.com/xt67).
