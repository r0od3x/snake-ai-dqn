# Snake AI — Deep Q-Learning

An agent that teaches itself to play **Snake** with **Deep Q-Learning** (reinforcement learning),
built with PyTorch and Pygame. It starts by moving randomly and, game after game, learns to chase the
food and avoid walls and its own tail, with no hard-coded strategy.

## How it works

The game loop is: **observe state → choose action → get reward → learn**.

**State (14 values, all 0/1)**

| Group | Values |
|---|---|
| Danger 1 cell ahead | straight, right, left |
| Danger 2 cells ahead | straight, right, left |
| Current direction | left, right, up, down |
| Food position | left, right, up, down (relative to the head) |

**Actions (3)** — relative to the current direction: `[straight, turn right, turn left]`

**Rewards** — `+10` for eating food, `-10` for dying (wall, own body, or looping for too long), `0` otherwise.

**Model** — a small fully-connected network `14 → 256 (ReLU) → 3` that predicts the Q-value of each action.

**Training**
- Bellman update: `Q_new = reward + γ · max(Q(next_state))`, MSE loss, Adam optimiser
- **Short-term memory**: trains on every step
- **Experience replay**: after each game, trains on a random batch of up to 1,000 past transitions
  (memory of 100,000)
- **ε-greedy exploration**: random moves are frequent at first and fade out over the first 80 games
- The model is saved to `model/model.pth` every time a new record score is reached
  (the shipped `model/pretrained.pth` reached an average of ~30 points per game)

## Getting started

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

**Watch the pre-trained agent play** (uses `model/pretrained.pth`, no exploration, no training):

```bash
python agent.py --play            # unlimited games
python agent.py --play --games 10
```

**Train your own agent from scratch** (opens the game window and a live score chart).
Each new record is saved to `model/model.pth`, the pre-trained model is never overwritten:

```bash
python agent.py
```

**Watch your own trained agent:**

```bash
python agent.py --play --model model.pth
```

## Hyperparameters

Defined at the top of the files and easy to tweak:

| Name | File | Value |
|---|---|---|
| `LR` | agent.py | 0.001 |
| `GAMMA` | agent.py | 0.9 |
| `MAX_MEMORY` | agent.py | 100,000 |
| `BATCH_SIZE` | agent.py | 1,000 |
| `EXPLORATION_GAMES` | agent.py | 80 |
| `SPEED` (FPS while training) | game.py | 60 |
| `BLOCK_SIZE` | game.py | 20 px |

## Project structure

```
├── agent.py        # Agent: state encoding, ε-greedy policy, memory, train / play loops
├── game.py         # Snake environment (Pygame) with a step(action) → reward API
├── model.py        # Linear_QNet (PyTorch) and the Q-learning trainer
├── helper.py       # Live training chart (matplotlib)
├── model/pretrained.pth  # Pre-trained weights
└── requirements.txt
```
