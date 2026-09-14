"""
GridWorld environment compatible with Gymnasium.

A 2D grid with obstacles where an agent navigates from a start position
to a goal. Supports 4-directional (cardinal) and 8-directional (with
diagonals) movement, optional reward shaping, and BFS solvability checks.

Observation space: Discrete(width * height)  — flat integer index
Action space:      Discrete(4) or Discrete(8)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    2D GridWorld with obstacles, configurable movement, and reward shaping.

    The agent observes a flat integer state (row * width + col) and selects
    integer actions. This keeps the environment compatible with any tabular
    agent that works with Discrete observation/action spaces.

    Key features ported from prior projects:
    - Obstacle generation: random scatter or clustered (from Python project)
    - BFS solvability guarantee (from C++ project)
    - max_steps truncation with terminated/truncated split (from C++ project)
    - Potential-based reward shaping (from Python project)
    """

    metadata = {"render_modes": ["ansi"]}

    # Action deltas: (row_delta, col_delta)
    ACTIONS_4: list[tuple[int, int]] = [
        (-1, 0),   # UP
        (1, 0),    # DOWN
        (0, -1),   # LEFT
        (0, 1),    # RIGHT
    ]
    ACTIONS_8: list[tuple[int, int]] = ACTIONS_4 + [
        (-1, -1),  # UP-LEFT
        (-1, 1),   # UP-RIGHT
        (1, -1),   # DOWN-LEFT
        (1, 1),    # DOWN-RIGHT
    ]

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        obstacle_density: float = 0.2,
        obstacle_mode: str = "cluster",
        seed: int | None = None,
        cluster_size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
        allow_diagonal: bool = False,
        diagonal_cost: float | None = None,
        reward_goal: float = 100.0,
        reward_obstacle: float = -10.0,
        reward_step: float = -0.01,
        shaping: str | None = None,
        max_steps: int = 500,
        ensure_solvable: bool = True,
        render_mode: str | None = None,
    ):
        """
        Create a GridWorld environment.

        Args:
            width: Grid width in cells.
            height: Grid height in cells.
            obstacle_density: Fraction of cells that are obstacles (0.0 to 1.0).
            obstacle_mode: How to place obstacles — "random" or "cluster".
            seed: Random seed for obstacle generation.
            cluster_size: Controls cluster spread when obstacle_mode="cluster".
            start: (row, col) of the start cell.
            goal: (row, col) of the goal cell. Defaults to bottom-right corner.
            allow_diagonal: If True, enable 8-directional movement.
            diagonal_cost: Multiplier for diagonal step penalty.
                           Defaults to sqrt(2) if allow_diagonal, else 1.0.
            reward_goal: Reward for reaching the goal (episode terminates).
            reward_obstacle: Penalty for hitting a wall or obstacle (agent stays).
            reward_step: Penalty per step (cardinal move).
            shaping: Potential-based reward shaping — None, "euclidean", or
                     "manhattan". Adds F = Φ(s) - Φ(s') where Φ is the distance
                     to the goal.
            max_steps: Maximum steps before the episode is truncated.
            ensure_solvable: If True, regenerate obstacles until a path exists.
            render_mode: Gymnasium render mode — "ansi" or None.
        """
        super().__init__()

        # Store configuration
        self._width = width
        self._height = height
        self._obstacle_density = obstacle_density
        self._obstacle_mode = obstacle_mode
        self._cluster_size = cluster_size
        self._start = start
        self._goal = goal if goal is not None else (height - 1, width - 1)
        self._allow_diagonal = allow_diagonal
        self._diagonal_cost = (
            diagonal_cost
            if diagonal_cost is not None
            else (math.sqrt(2) if allow_diagonal else 1.0)
        )
        self._reward_goal = reward_goal
        self._reward_obstacle = reward_obstacle
        self._reward_step = reward_step
        self._shaping = shaping
        self._max_steps = max_steps
        self._ensure_solvable = ensure_solvable
        self.render_mode = render_mode

        # Action set
        self._actions = self.ACTIONS_8 if allow_diagonal else self.ACTIONS_4

        # Gymnasium spaces
        self.observation_space = spaces.Discrete(width * height)
        self.action_space = spaces.Discrete(len(self._actions))

        # Build the grid
        self._rng = np.random.default_rng(seed)
        self._grid = self._build_valid_grid(seed)

        # Pre-compute potential map for reward shaping
        self._potential_map: np.ndarray | None = None
        if self._shaping is not None:
            self._potential_map = self._compute_potential_map()

        # Episode state (initialized properly in reset())
        self._agent_pos: tuple[int, int] = self._start
        self._steps_taken: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset the environment to the start state.

        Returns:
            obs: Flat integer state index.
            info: Dictionary with agent position and step count.
        """
        super().reset(seed=seed)
        self._agent_pos = self._start
        self._steps_taken = 0
        return self._rc_to_state(*self._agent_pos), self._get_info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Take one step in the environment.

        Args:
            action: Integer action index.

        Returns:
            obs: Next state as flat integer.
            reward: Scalar reward.
            terminated: True if the agent reached the goal.
            truncated: True if the episode exceeded max_steps.
            info: Dictionary with agent position and step count.
        """
        r, c = self._agent_pos
        dr, dc = self._actions[action]
        nr, nc = r + dr, c + dc

        # Check if it's an illegal diagonal squeeze (corner cutting)
        illegal_diagonal = False
        if self._allow_diagonal and dr != 0 and dc != 0:
            w1 = not self._in_bounds(r + dr, c) or self._grid[r + dr, c] == 1
            w2 = not self._in_bounds(r, c + dc) or self._grid[r, c + dc] == 1
            if w1 and w2:
                illegal_diagonal = True

        # Check bounds and obstacles
        hit_obstacle = False
        if not self._in_bounds(nr, nc) or self._grid[nr, nc] == 1 or illegal_diagonal:
            # Invalid move: agent stays in place, receives obstacle penalty
            next_pos = (r, c)
            reward = self._reward_obstacle
            terminated = False
            hit_obstacle = True
        elif (nr, nc) == self._goal:
            # Reached the goal
            next_pos = (nr, nc)
            reward = self._reward_goal
            terminated = True
        else:
            # Valid move to a free cell
            next_pos = (nr, nc)
            terminated = False
            # Diagonal moves cost more (proportional to distance traveled)
            if self._allow_diagonal and dr != 0 and dc != 0:
                reward = self._reward_step * self._diagonal_cost
            else:
                reward = self._reward_step

        # Potential-based reward shaping: F = Φ(s) - Φ(s')
        # Encourages moving closer to the goal without changing the optimal policy
        if self._potential_map is not None and not terminated:
            reward += self._potential_map[r, c] - self._potential_map[next_pos[0], next_pos[1]]

        # Update state
        self._agent_pos = next_pos
        self._steps_taken += 1

        # Truncation: episode exceeded max_steps without reaching goal
        truncated = (not terminated) and (self._steps_taken >= self._max_steps)

        obs = self._rc_to_state(*self._agent_pos)
        info = self._get_info()
        info["hit_obstacle"] = hit_obstacle
        return obs, float(reward), terminated, truncated, info

    def render(self) -> str | None:
        """Render the grid as ASCII text."""
        if self.render_mode != "ansi":
            return None

        lines: list[str] = []
        for r in range(self._height):
            row: list[str] = []
            for c in range(self._width):
                if (r, c) == self._agent_pos:
                    row.append("A")
                elif (r, c) == self._start:
                    row.append("S")
                elif (r, c) == self._goal:
                    row.append("G")
                elif self._grid[r, c] == 1:
                    row.append("#")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public properties (for notebooks and visualization)
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Return the obstacle grid (height, width) — 0=free, 1=obstacle."""
        return self._grid.copy()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def start(self) -> tuple[int, int]:
        """Start position as (row, col)."""
        return self._start

    @property
    def goal(self) -> tuple[int, int]:
        """Goal position as (row, col)."""
        return self._goal

    @property
    def actions(self) -> list[tuple[int, int]]:
        """List of action deltas as (row_delta, col_delta)."""
        return list(self._actions)

    @property
    def allow_diagonal(self) -> bool:
        return self._allow_diagonal

    # ------------------------------------------------------------------
    # Coordinate conversion (public for notebooks)
    # ------------------------------------------------------------------

    def state_to_rc(self, state: int) -> tuple[int, int]:
        """Convert flat state index to (row, col)."""
        return divmod(state, self._width)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rc_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) to flat state index."""
        return row * self._width + col

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if (row, col) is within the grid."""
        return 0 <= row < self._height and 0 <= col < self._width

    def _get_info(self) -> dict[str, Any]:
        """Build the info dictionary returned by reset/step."""
        return {
            "agent_pos": self._agent_pos,
            "steps_taken": self._steps_taken,
        }

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    def _build_valid_grid(self, seed: int | None) -> np.ndarray:
        """Generate an obstacle grid, retrying until solvable if required.

        This implements the solvability guarantee from the C++ project:
        generate obstacles, check BFS reachability, retry if needed.
        """
        max_attempts = 100

        for attempt in range(max_attempts):
            grid = np.zeros((self._height, self._width), dtype=np.int8)
            self._populate_obstacles(grid)

            # Ensure start and goal are always free
            grid[self._start[0], self._start[1]] = 0
            grid[self._goal[0], self._goal[1]] = 0

            if not self._ensure_solvable or self._has_path(grid, self._start, self._goal):
                return grid

            # Retry with a different RNG state
            if seed is not None:
                self._rng = np.random.default_rng(seed + attempt + 1)
            # If seed is None, _rng already has a different state from the failed attempt

        raise RuntimeError(
            f"Could not generate a solvable grid after {max_attempts} attempts. "
            f"Try reducing obstacle_density={self._obstacle_density}."
        )

    def _populate_obstacles(self, grid: np.ndarray) -> None:
        """Place obstacles on the grid using the configured mode."""
        total_cells = self._width * self._height
        num_obstacles = int(total_cells * self._obstacle_density)
        if num_obstacles == 0:
            return

        if self._obstacle_mode == "random":
            indices = self._rng.choice(total_cells, size=num_obstacles, replace=False)
            grid.flat[indices] = 1

        elif self._obstacle_mode == "cluster":
            placed = 0
            while placed < num_obstacles:
                # Pick a random cluster center
                center_r = int(self._rng.integers(0, self._height))
                center_c = int(self._rng.integers(0, self._width))
                # Place obstacles near the center
                for _ in range(self._cluster_size):
                    if placed >= num_obstacles:
                        break
                    dr = int(self._rng.integers(-self._cluster_size, self._cluster_size + 1))
                    dc = int(self._rng.integers(-self._cluster_size, self._cluster_size + 1))
                    r, c = center_r + dr, center_c + dc
                    if self._in_bounds(r, c) and grid[r, c] == 0:
                        grid[r, c] = 1
                        placed += 1
        else:
            raise ValueError(
                f"Unknown obstacle_mode: {self._obstacle_mode!r}. "
                f"Expected 'random' or 'cluster'."
            )

    def _has_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> bool:
        """BFS to check if a path exists from start to goal.

        Uses the environment's action set (4 or 8 directions) so the
        check matches the agent's movement capabilities.
        """
        if start == goal:
            return True

        visited: set[tuple[int, int]] = {start}
        queue: deque[tuple[int, int]] = deque([start])

        while queue:
            r, c = queue.popleft()
            for dr, dc in self._actions:
                nr, nc = r + dr, c + dc
                
                # Check for illegal diagonal squeeze (corner cutting)
                illegal_diagonal = False
                if self._allow_diagonal and dr != 0 and dc != 0:
                    w1 = not self._in_bounds(r + dr, c) or grid[r + dr, c] == 1
                    w2 = not self._in_bounds(r, c + dc) or grid[r, c + dc] == 1
                    if w1 and w2:
                        illegal_diagonal = True

                if (
                    self._in_bounds(nr, nc)
                    and (nr, nc) not in visited
                    and grid[nr, nc] == 0
                    and not illegal_diagonal
                ):
                    if (nr, nc) == goal:
                        return True
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return False

    def _compute_potential_map(self) -> np.ndarray:
        """Pre-compute distance-to-goal for each cell (for reward shaping).

        Φ(s) = distance(s, goal), so moving closer to the goal gives
        positive shaping reward: F = Φ(s) - Φ(s') > 0 when getting closer.
        """
        potential = np.zeros((self._height, self._width))
        gr, gc = self._goal

        for r in range(self._height):
            for c in range(self._width):
                if self._shaping == "manhattan":
                    potential[r, c] = abs(r - gr) + abs(c - gc)
                elif self._shaping == "euclidean":
                    potential[r, c] = math.hypot(r - gr, c - gc)
                else:
                    raise ValueError(
                        f"Unknown shaping: {self._shaping!r}. "
                        f"Expected None, 'euclidean', or 'manhattan'."
                    )

        return potential
