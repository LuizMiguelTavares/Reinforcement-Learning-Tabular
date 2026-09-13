"""
CLI entry point for training a tabular RL agent.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from rl_tabular.agents import QLearningAgent
from rl_tabular.envs.gridworld import GridWorldEnv
from rl_tabular.training.runner import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular RL agent")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    args = parser.parse_args()

    # ---- Load config ----
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_cfg = config["environment"]
    agent_cfg = config["agent"]
    train_cfg = config["training"]
    run_name = config.get("run_name", "default_run")

    # ---- Create environment ----
    env = GridWorldEnv(
        width=env_cfg.get("width", 10),
        height=env_cfg.get("height", 10),
        obstacle_density=env_cfg.get("obstacle_density", 0.2),
        obstacle_mode=env_cfg.get("obstacle_mode", "cluster"),
        seed=env_cfg.get("seed"),
        cluster_size=env_cfg.get("cluster_size", 5),
        allow_diagonal=env_cfg.get("allow_diagonal", False),
        reward_goal=env_cfg.get("reward_goal", 100.0),
        reward_obstacle=env_cfg.get("reward_obstacle", -10.0),
        reward_step=env_cfg.get("reward_step", -0.01),
        shaping=env_cfg.get("shaping"),
        max_steps=env_cfg.get("max_steps", 500),
        ensure_solvable=env_cfg.get("ensure_solvable", True),
    )

    print(f"GridWorld: {env.width}×{env.height}, "
          f"obstacles={env.grid.sum()}/{env.width * env.height}, "
          f"actions={env.action_space.n}")
    print(f"Start: {env.start}, Goal: {env.goal}")

    # ---- Create agent ----
    agent = QLearningAgent(
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
        alpha=agent_cfg.get("alpha", 0.1),
        gamma=agent_cfg.get("gamma", 0.99),
        seed=agent_cfg.get("seed"),
    )

    # ---- Run training ----
    run_dir = Path("runs") / run_name
    resume = args.resume or train_cfg.get("resume", False)

    # Save a copy of the config for reproducibility
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")

    train(
        env=env,
        agent=agent,
        num_episodes=train_cfg.get("num_episodes", 50000),
        run_dir=run_dir,
        epsilon_start=train_cfg.get("epsilon_start", 1.0),
        epsilon_end=train_cfg.get("epsilon_end", 0.05),
        checkpoint_every=train_cfg.get("checkpoint_every", 10000),
        print_every=train_cfg.get("print_every", 1000),
        flush_every=train_cfg.get("flush_every", 100),
        resume=resume,
    )

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
