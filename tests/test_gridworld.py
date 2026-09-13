"""Tests for the GridWorld environment."""

from __future__ import annotations

import numpy as np
import pytest

from rl_tabular.envs.gridworld import GridWorldEnv


class TestGridWorldCreation:
    """Test environment construction and obstacle generation."""

    def test_creates_with_defaults(self) -> None:
        env = GridWorldEnv()
        assert env.width == 10
        assert env.height == 10
        assert env.observation_space.n == 100
        assert env.action_space.n == 4

    def test_creates_with_diagonal(self) -> None:
        env = GridWorldEnv(allow_diagonal=True)
        assert env.action_space.n == 8

    def test_start_and_goal_are_free(self) -> None:
        env = GridWorldEnv(obstacle_density=0.3, seed=42)
        grid = env.grid
        assert grid[env.start[0], env.start[1]] == 0
        assert grid[env.goal[0], env.goal[1]] == 0

    def test_custom_start_and_goal(self) -> None:
        env = GridWorldEnv(
            width=5, height=5, obstacle_density=0.0,
            start=(1, 1), goal=(3, 3),
        )
        assert env.start == (1, 1)
        assert env.goal == (3, 3)

    def test_default_goal_is_bottom_right(self) -> None:
        env = GridWorldEnv(width=7, height=5, obstacle_density=0.0)
        assert env.goal == (4, 6)  # (height-1, width-1)

    def test_ensure_solvable_generates_valid_grid(self) -> None:
        # High density, but solvability is guaranteed
        env = GridWorldEnv(
            obstacle_density=0.3, seed=42, ensure_solvable=True,
        )
        # Smoke test: environment was created without error
        assert env.grid.sum() > 0

    def test_no_obstacles(self) -> None:
        env = GridWorldEnv(obstacle_density=0.0)
        assert env.grid.sum() == 0

    def test_random_obstacle_mode(self) -> None:
        env = GridWorldEnv(obstacle_mode="random", seed=42, obstacle_density=0.1)
        assert env.grid.sum() > 0

    def test_cluster_obstacle_mode(self) -> None:
        env = GridWorldEnv(obstacle_mode="cluster", seed=42, obstacle_density=0.1)
        assert env.grid.sum() > 0

    def test_invalid_obstacle_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown obstacle_mode"):
            GridWorldEnv(obstacle_mode="invalid")

    def test_seed_reproducibility(self) -> None:
        env1 = GridWorldEnv(seed=123, obstacle_density=0.2)
        env2 = GridWorldEnv(seed=123, obstacle_density=0.2)
        np.testing.assert_array_equal(env1.grid, env2.grid)


class TestGridWorldAPI:
    """Test the Gymnasium API compliance."""

    def test_reset_returns_int_and_dict(self) -> None:
        env = GridWorldEnv()
        obs, info = env.reset()
        assert isinstance(obs, (int, np.integer))
        assert isinstance(info, dict)

    def test_step_returns_5_tuple(self) -> None:
        env = GridWorldEnv()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, (int, np.integer))
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_in_observation_space(self) -> None:
        env = GridWorldEnv()
        obs, _ = env.reset()
        assert env.observation_space.contains(int(obs))

    def test_reset_returns_start_state(self) -> None:
        env = GridWorldEnv(width=5, height=5, obstacle_density=0.0)
        obs, _ = env.reset()
        r, c = env.state_to_rc(obs)
        assert (r, c) == env.start

    def test_reaching_goal_terminates(self) -> None:
        # Small empty grid: start=(0,0), goal=(1,1)
        env = GridWorldEnv(
            width=2, height=2, obstacle_density=0.0, max_steps=100,
        )
        obs, _ = env.reset()
        # DOWN: (0,0) -> (1,0)
        obs, _, terminated, _, _ = env.step(1)
        assert not terminated
        # RIGHT: (1,0) -> (1,1) = goal
        obs, _, terminated, _, _ = env.step(3)
        assert terminated

    def test_max_steps_causes_truncation(self) -> None:
        env = GridWorldEnv(
            width=5, height=5, obstacle_density=0.0, max_steps=3,
        )
        env.reset()
        # Take 3 steps hitting walls (agent stays in place)
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(0)  # UP from (0,0) = wall
            if i < 2:
                assert not truncated
        # After max_steps, should be truncated
        assert truncated or terminated

    def test_hitting_wall_stays_in_place(self) -> None:
        env = GridWorldEnv(
            width=3, height=3, obstacle_density=0.0, max_steps=100,
        )
        obs, _ = env.reset()  # start at (0,0)
        # Move UP from top row: should stay in place
        obs_next, _, _, _, info = env.step(0)
        assert info["agent_pos"] == (0, 0)

    def test_hitting_obstacle_stays_in_place(self) -> None:
        env = GridWorldEnv(
            width=3, height=3, obstacle_density=0.0, max_steps=100,
        )
        # Manually place an obstacle at (1, 0)
        env._grid[1, 0] = 1
        env.reset()
        # Move DOWN from (0,0) into obstacle at (1,0)
        _, reward, _, _, info = env.step(1)
        assert info["agent_pos"] == (0, 0)
        assert reward == env._reward_obstacle


class TestGridWorldStateConversion:
    """Test state ↔ (row, col) conversion."""

    def test_rc_to_state(self) -> None:
        env = GridWorldEnv(width=5, height=3)
        # (0,0) -> 0, (0,4) -> 4, (1,0) -> 5, (2,4) -> 14
        assert env._rc_to_state(0, 0) == 0
        assert env._rc_to_state(0, 4) == 4
        assert env._rc_to_state(1, 0) == 5
        assert env._rc_to_state(2, 4) == 14

    def test_state_to_rc(self) -> None:
        env = GridWorldEnv(width=5, height=3)
        assert env.state_to_rc(0) == (0, 0)
        assert env.state_to_rc(4) == (0, 4)
        assert env.state_to_rc(5) == (1, 0)
        assert env.state_to_rc(14) == (2, 4)

    def test_roundtrip(self) -> None:
        env = GridWorldEnv(width=7, height=5)
        for r in range(env.height):
            for c in range(env.width):
                state = env._rc_to_state(r, c)
                assert env.state_to_rc(state) == (r, c)


class TestGridWorldRender:
    """Test ASCII rendering."""

    def test_ansi_render(self) -> None:
        env = GridWorldEnv(
            width=3, height=3, obstacle_density=0.0,
            render_mode="ansi",
        )
        env.reset()
        output = env.render()
        assert output is not None
        assert "A" in output  # agent (at start)
        assert "G" in output  # goal

    def test_no_render_without_mode(self) -> None:
        env = GridWorldEnv(width=3, height=3, obstacle_density=0.0)
        env.reset()
        assert env.render() is None
