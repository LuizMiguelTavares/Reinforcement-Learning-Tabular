"""Tests for tabular RL agents."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from rl_tabular.agents.base import TabularAgent
from rl_tabular.agents.td import QLearningAgent


class TestTabularAgent:
    """Test the TabularAgent base class."""

    def test_q_table_shape(self) -> None:
        agent = TabularAgent(num_states=10, num_actions=4)
        assert agent.Q.shape == (10, 4)
        assert np.all(agent.Q == 0.0)

    def test_greedy_action_picks_best(self) -> None:
        agent = TabularAgent(num_states=4, num_actions=3, seed=42)
        agent.Q[0] = [1.0, 5.0, 2.0]
        assert agent.greedy_action(0) == 1

    def test_greedy_action_tie_breaking(self) -> None:
        agent = TabularAgent(num_states=4, num_actions=3, seed=42)
        agent.Q[0] = [5.0, 5.0, 5.0]  # all equal
        # Should not crash, and should return a valid action
        action = agent.greedy_action(0)
        assert 0 <= action < 3

    def test_epsilon_greedy_explores_at_epsilon_1(self) -> None:
        agent = TabularAgent(
            num_states=4, num_actions=3, epsilon=1.0, seed=42,
        )
        agent.Q[0] = [0.0, 100.0, 0.0]
        # With ε=1.0, should explore (not always pick action 1)
        actions = [agent.epsilon_greedy_action(0) for _ in range(100)]
        assert len(set(actions)) > 1

    def test_epsilon_greedy_exploits_at_epsilon_0(self) -> None:
        agent = TabularAgent(
            num_states=4, num_actions=3, epsilon=0.0, seed=42,
        )
        agent.Q[0] = [0.0, 100.0, 0.0]
        # With ε=0.0, should always pick action 1
        actions = [agent.epsilon_greedy_action(0) for _ in range(100)]
        assert all(a == 1 for a in actions)

    def test_value(self) -> None:
        agent = TabularAgent(num_states=4, num_actions=3)
        agent.Q[2] = [1.0, 5.0, 3.0]
        assert agent.value(2) == 5.0

    def test_value_initial_is_zero(self) -> None:
        agent = TabularAgent(num_states=4, num_actions=3)
        assert agent.value(0) == 0.0

    def test_save_load_roundtrip(self) -> None:
        agent = TabularAgent(
            num_states=4, num_actions=3, epsilon=0.42, seed=42,
        )
        agent.Q[1, 2] = 99.0
        agent.Q[3, 0] = -5.5

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)

            # Create a new agent and load
            agent2 = TabularAgent(num_states=4, num_actions=3)
            agent2.load(tmpdir)

            np.testing.assert_array_equal(agent2.Q, agent.Q)
            assert agent2.epsilon == 0.42

    def test_save_creates_files(self) -> None:
        agent = TabularAgent(num_states=4, num_actions=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)
            from pathlib import Path

            assert (Path(tmpdir) / "q_table.npz").exists()
            assert (Path(tmpdir) / "metadata.json").exists()


class TestQLearningAgent:
    """Test the Q-Learning update rule."""

    def test_inherits_from_tabular(self) -> None:
        agent = QLearningAgent(num_states=4, num_actions=2)
        assert isinstance(agent, TabularAgent)

    def test_has_alpha(self) -> None:
        agent = QLearningAgent(num_states=4, num_actions=2, alpha=0.5)
        assert agent.alpha == 0.5

    def test_update_from_zero(self) -> None:
        """Test update with Q-table initialized to zero."""
        agent = QLearningAgent(
            num_states=4, num_actions=2, alpha=0.1, gamma=0.9,
        )
        # Q initially all zeros
        # update(s=0, a=0, r=1.0, s_next=1, terminated=False)
        # target = 1.0 + 0.9 * max(Q[1]) = 1.0 + 0.9 * 0 = 1.0
        # td_error = 1.0 - 0.0 = 1.0
        # Q[0,0] = 0 + 0.1 * 1.0 = 0.1
        agent.update(0, 0, 1.0, 1, False)
        assert abs(agent.Q[0, 0] - 0.1) < 1e-10

    def test_update_terminal(self) -> None:
        """When terminated, there's no bootstrap: target = r."""
        agent = QLearningAgent(
            num_states=4, num_actions=2, alpha=0.5, gamma=0.9,
        )
        # target = 10.0 (no bootstrap because terminated)
        # Q[0,0] = 0 + 0.5 * (10.0 - 0) = 5.0
        agent.update(0, 0, 10.0, 1, True)
        assert abs(agent.Q[0, 0] - 5.0) < 1e-10

    def test_update_with_existing_q(self) -> None:
        """Test update when Q-table already has values."""
        agent = QLearningAgent(
            num_states=4, num_actions=2, alpha=0.1, gamma=0.9,
        )
        agent.Q[0, 0] = 2.0
        agent.Q[1, 0] = 5.0
        agent.Q[1, 1] = 3.0

        # update(s=0, a=0, r=1.0, s_next=1, terminated=False)
        # target = 1.0 + 0.9 * max(Q[1]) = 1.0 + 0.9 * 5.0 = 5.5
        # td_error = 5.5 - 2.0 = 3.5
        # Q[0,0] = 2.0 + 0.1 * 3.5 = 2.35
        agent.update(0, 0, 1.0, 1, False)
        assert abs(agent.Q[0, 0] - 2.35) < 1e-10

    def test_update_negative_reward(self) -> None:
        """Test update with negative reward (penalty)."""
        agent = QLearningAgent(
            num_states=4, num_actions=2, alpha=1.0, gamma=0.5,
        )
        # alpha=1.0 means Q is fully replaced by target
        # target = -10.0 + 0.5 * max(Q[2]) = -10.0 + 0 = -10.0
        # Q[0,0] = 0 + 1.0 * (-10.0 - 0) = -10.0
        agent.update(0, 0, -10.0, 2, False)
        assert abs(agent.Q[0, 0] - (-10.0)) < 1e-10

    def test_multiple_updates_converge(self) -> None:
        """Verify that repeated updates move Q toward the target."""
        agent = QLearningAgent(
            num_states=2, num_actions=1, alpha=0.1, gamma=0.0,
        )
        # gamma=0: target is always just r=1.0
        # Repeated updates should converge Q[0,0] toward 1.0
        for _ in range(100):
            agent.update(0, 0, 1.0, 1, False)
        assert abs(agent.Q[0, 0] - 1.0) < 0.01

    def test_q_learning_is_off_policy(self) -> None:
        """Q-Learning uses max over next-state Q-values (off-policy).

        This test verifies that the update target uses max_a Q(s',a')
        regardless of what action was actually taken in s'.
        """
        agent = QLearningAgent(
            num_states=4, num_actions=3, alpha=1.0, gamma=1.0,
        )
        agent.Q[2, 0] = 10.0  # best action in state 2
        agent.Q[2, 1] = 1.0
        agent.Q[2, 2] = 5.0

        # The target should use max(Q[2]) = 10.0, not any specific action
        # target = 0.0 + 1.0 * 10.0 = 10.0
        agent.update(0, 0, 0.0, 2, False)
        assert abs(agent.Q[0, 0] - 10.0) < 1e-10
