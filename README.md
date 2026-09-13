# Reinforcement-Learning-Tabular

Tabular RL algorithms from **Sutton & Barto (Part 1)** — a study project for understanding the fundamentals of reinforcement learning.

## Algorithms

| Chapter | Algorithm | Status |
|---------|-----------|--------|
| 6 | Q-Learning | ✅ |
| 6 | SARSA | 🔲 |
| 6 | Expected SARSA | 🔲 |
| 5 | Monte Carlo | 🔲 |
| 7 | n-step TD | 🔲 |
| 8 | Dyna-Q | 🔲 |
| 4 | Dynamic Programming | 🔲 |

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install in editable mode (with dev dependencies)
pip install -e ".[dev]"

# 3. Run training
python scripts/train.py --config configs/default.yaml

# 4. Run tests
pytest
```

## Project Structure

```
src/rl_tabular/
├── envs/           # Gymnasium-compatible environments
│   └── gridworld.py
├── agents/         # Tabular RL agents (decoupled from environments)
│   ├── base.py     # TabularAgent base class
│   └── td.py       # Q-Learning (SARSA, Expected SARSA — TODO)
├── training/       # Training loops, logging, checkpointing
│   ├── runner.py
│   └── logger.py
└── utils/          # Visualization helpers for Jupyter notebooks
    └── visualization.py
```

## Usage

```python
from rl_tabular.envs import GridWorldEnv
from rl_tabular.agents import QLearningAgent
from rl_tabular.training import train

env = GridWorldEnv(width=10, height=10, obstacle_density=0.2, seed=42)
agent = QLearningAgent(
    num_states=env.observation_space.n,
    num_actions=env.action_space.n,
    alpha=0.1, gamma=0.99,
)

train(env, agent, num_episodes=50_000, run_dir="runs/my_experiment")
```

## Training Outputs

```
runs/<run_name>/
├── train_history.csv       # Per-episode metrics
├── config.yaml             # Copy of the config used
└── checkpoints/latest/     # Q-table + metadata (for resume)
```

## Analysis

Open a Jupyter notebook and use the built-in visualization tools:

```python
from rl_tabular.utils.visualization import plot_value_and_policy, plot_learning_curve

plot_value_and_policy(env, agent)
plot_learning_curve("runs/my_experiment/train_history.csv")
```
