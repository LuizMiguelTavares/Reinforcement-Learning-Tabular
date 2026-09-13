"""
Base class for all tabular RL agents.

Provides the shared infrastructure that every tabular agent needs:
- Q-table as a 2D numpy array (num_states × num_actions)
- Epsilon-greedy and greedy action selection
- Value function V(s) = max_a Q(s,a)
- Save/load to disk

NOTE: This class does NOT define an abstract update() method because
different agent families have fundamentally different update signatures:
- TD agents update per step:    update(s, a, r, s', terminated)
- MC agents update per episode: update_from_episode(episode)
- DP agents sweep all states:   policy_evaluation_sweep(env)
Forcing a single signature would be bad design.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TabularAgent:
    """Base class for tabular RL agents with a Q-table."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize the agent.

        Args:
            num_states: Size of the state space |S|.
            num_actions: Size of the action space |A|.
            gamma: Discount factor.
            epsilon: Initial exploration rate for ε-greedy.
            seed: Random seed for reproducibility.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Q-table: always 2D (|S|, |A|), initialized to zero
        self.Q = np.zeros((num_states, num_actions))

    def greedy_action(self, state: int) -> int:
        """Return the greedy action: argmax_a Q(s, a).

        Ties are broken randomly to avoid directional bias.
        """
        q_values = self.Q[state]
        max_q = q_values.max()
        best_actions = np.flatnonzero(q_values == max_q)
        return int(self.rng.choice(best_actions))

    def epsilon_greedy_action(self, state: int) -> int:
        """ε-greedy action selection.

        With probability ε, choose a random action (explore).
        With probability 1-ε, choose the greedy action (exploit).
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        return self.greedy_action(state)

    def value(self, state: int) -> float:
        """State value: V(s) = max_a Q(s, a)."""
        return float(self.Q[state].max())

    def save(self, path: str | Path) -> None:
        """Save Q-table and agent metadata to a directory.

        Creates:
            path/q_table.npz  — compressed Q-table
            path/metadata.json — agent hyperparameters
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(path / "q_table.npz", Q=self.Q)

        metadata = {
            "num_states": int(self.num_states),
            "num_actions": int(self.num_actions),
            "gamma": float(self.gamma),
            "epsilon": float(self.epsilon),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load Q-table and metadata from a directory.

        Restores the Q-table and epsilon value. The agent's dimensions
        (num_states, num_actions) must match the saved data.
        """
        path = Path(path)

        data = np.load(path / "q_table.npz")
        self.Q = data["Q"]

        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        self.epsilon = metadata["epsilon"]
