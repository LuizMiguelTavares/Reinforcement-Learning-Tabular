"""
Visualization utilities for GridWorld and tabular agents.

All functions are designed to be called from Jupyter notebooks. They accept
an optional matplotlib Axes to allow flexible composition:

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_value_heatmap(env, agent, ax=axes[0])
    plot_policy_arrows(env, agent, ax=axes[0])  # overlay on same axes
    plot_greedy_path(env, agent, ax=axes[1])
    plot_learning_curve("runs/my_run/train_history.csv", ax=axes[2])
    plt.tight_layout()
    plt.show()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_tabular.agents.base import TabularAgent
from rl_tabular.envs.gridworld import GridWorldEnv


def greedy_rollout(
    env: GridWorldEnv,
    agent: TabularAgent,
    max_steps: int = 1000,
) -> list[int]:
    """Execute the greedy policy and return a list of visited states.

    Args:
        env: The GridWorld environment.
        agent: A trained tabular agent.
        max_steps: Maximum steps to prevent infinite loops.

    Returns:
        List of flat state indices visited during the rollout.
    """
    obs, _ = env.reset()
    path = [obs]
    for _ in range(max_steps):
        action = agent.greedy_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        path.append(obs)
        if terminated or truncated:
            break
    return path


def plot_value_heatmap(
    env: GridWorldEnv,
    agent: TabularAgent,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot V(s) = max_a Q(s,a) as a color heatmap.

    Obstacles are shown in black. Start and goal are marked.
    Ported from the combined_vis() in the original Python project.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    grid = env.grid

    # Build value map
    V = np.zeros((env.height, env.width))
    for r in range(env.height):
        for c in range(env.width):
            state = r * env.width + c
            V[r, c] = agent.value(state)

    # Mask obstacles so they don't affect the colormap range
    V_masked = np.ma.masked_where(grid == 1, V)

    cmap = "turbo" if "turbo" in plt.colormaps() else "plasma"
    im = ax.imshow(V_masked, cmap=cmap, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Draw obstacles in black
    obs_masked = np.ma.masked_where(grid == 0, grid)
    ax.imshow(
        obs_masked,
        cmap="gray_r",
        origin="upper",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        alpha=1,
    )

    # Mark start and goal
    ax.scatter(
        env.start[1], env.start[0],
        marker="o", c="lime", s=100, zorder=5, label="Start",
    )
    ax.scatter(
        env.goal[1], env.goal[0],
        marker="*", c="red", s=150, zorder=5, label="Goal",
    )

    ax.set_title("State Values V(s)")
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def plot_policy_arrows(
    env: GridWorldEnv,
    agent: TabularAgent,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay greedy policy arrows on the current axes.

    Each free cell gets an arrow pointing in the direction of the
    greedy action argmax_a Q(s, a).
    Ported from the quiver plot in the original Python project.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    grid = env.grid
    actions_list = env.actions

    Y, X = np.mgrid[0 : env.height, 0 : env.width]
    U = np.zeros((env.height, env.width))
    V_arr = np.zeros((env.height, env.width))

    for r in range(env.height):
        for c in range(env.width):
            if grid[r, c] == 1:
                continue
            state = r * env.width + c
            best_action = agent.greedy_action(state)
            dr, dc = actions_list[best_action]
            U[r, c] = dc       # x-component
            V_arr[r, c] = -dr  # y-component (inverted for imshow coords)

    free = grid.flatten() == 0
    ax.quiver(
        X.flatten()[free],
        Y.flatten()[free],
        U.flatten()[free],
        V_arr.flatten()[free],
        color="white",
        scale=max(env.width, env.height) * 1.5,
        pivot="mid",
        headwidth=4,
        headlength=5,
        width=0.002,
    )

    return ax


def plot_value_and_policy(
    env: GridWorldEnv,
    agent: TabularAgent,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Combined plot: value heatmap + policy arrows.

    Equivalent to the combined_vis() from the original Python project.
    """
    ax = plot_value_heatmap(env, agent, ax=ax)
    plot_policy_arrows(env, agent, ax=ax)
    ax.set_title("State Values + Greedy Policy")
    return ax


def plot_greedy_path(
    env: GridWorldEnv,
    agent: TabularAgent,
    ax: plt.Axes | None = None,
    max_steps: int = 1000,
) -> plt.Axes:
    """Execute the greedy policy and draw the path on the grid.

    Free cells are white, obstacles are black, the path is blue,
    start is green, and goal is red.
    Ported from path_image() + greedy_path() in the original Python project.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    grid = env.grid
    path_states = greedy_rollout(env, agent, max_steps)

    # Build RGB image
    img = np.ones((env.height, env.width, 3))
    img[grid == 1] = (0.0, 0.0, 0.0)   # obstacles: black
    img[env.start[0], env.start[1]] = (0.0, 1.0, 0.0)  # start: green
    img[env.goal[0], env.goal[1]] = (1.0, 0.0, 0.0)    # goal: red

    # Color the path cells
    for state in path_states:
        r, c = env.state_to_rc(state)
        if (r, c) not in (env.start, env.goal) and grid[r, c] == 0:
            img[r, c] = (0.5, 0.5, 1.0)  # path: light blue

    ax.imshow(img, origin="upper")
    ax.set_title(f"Greedy Path (length={len(path_states)})")
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def plot_learning_curve(
    csv_path: str | Path,
    window: int = 100,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot rolling-average learning curve from the training CSV.

    Reads the CSV generated by CSVLogger and plots the rolling mean
    of episode_return. Requires pandas (optional dev dependency).
    """
    import pandas as pd

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    df = pd.read_csv(csv_path)
    rolling = df["episode_return"].rolling(window, min_periods=1).mean()

    ax.plot(df["episode"], rolling, linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Return (rolling avg, window={window})")
    ax.set_title("Learning Curve")
    ax.grid(True, alpha=0.3)

    return ax
