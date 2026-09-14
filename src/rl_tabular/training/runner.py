"""
Training loop with checkpointing, resume, and graceful Ctrl+C.

This module provides the main train() function for step-by-step TD agents
(Q-Learning, SARSA, Expected SARSA). Monte Carlo and DP agents will need
their own training loops, but they can reuse CSVLogger and the checkpoint
infrastructure.

Features ported from the C++ project:
- Periodic safety checkpoints (checkpoints/latest/)
- Resume from checkpoint (preserving episode count and epsilon)
- Graceful SIGINT handling (finish current episode, save, exit)
- Structured CSV logging for empirical analysis
"""

from __future__ import annotations

import json
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import gymnasium as gym

from rl_tabular.agents.base import TabularAgent
from rl_tabular.training.logger import CSVLogger


def train(
    env: gym.Env,
    agent: TabularAgent,
    num_episodes: int,
    run_dir: str | Path,
    *,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int | None = None,
    checkpoint_every: int = 10_000,
    print_every: int = 1000,
    flush_every: int = 100,
    resume: bool = False,
) -> Path:
    """
    Train a TD agent on an environment.

    This implements the standard online TD training loop:
        for each episode:
            for each step:
                action = agent.epsilon_greedy_action(obs)
                obs_next, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, action, reward, obs_next, terminated)

    The agent must implement update(s, a, r, s_next, terminated).

    Args:
        env: Gymnasium environment with Discrete observation/action spaces.
        agent: Tabular agent with update(s, a, r, s_next, terminated) method.
        num_episodes: Number of episodes to train in this execution.
        run_dir: Directory for logs, checkpoints, and config backup.
        epsilon_start: Initial ε value for exploration.
        epsilon_end: Final ε value.
        epsilon_decay_episodes: Episodes over which ε decays linearly.
                                Defaults to 80% of num_episodes.
        checkpoint_every: Save checkpoint every N episodes.
        print_every: Print progress every N episodes.
        flush_every: Flush CSV to disk every N episodes.
        resume: If True, load from checkpoints/latest/ and continue.

    Returns:
        Path to the run directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints" / "latest"

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_episode = 0
    if resume and (checkpoint_dir / "metadata.json").exists():
        agent.load(checkpoint_dir)
        with open(checkpoint_dir / "metadata.json") as f:
            meta = json.load(f)
        start_episode = meta.get("last_completed_episode", 0)
        print(
            f"[Resume] Loaded checkpoint at episode {start_episode}, "
            f"ε={agent.epsilon:.4f}"
        )
    else:
        agent.epsilon = epsilon_start

    # ------------------------------------------------------------------
    # Epsilon decay schedule (linear)
    # ------------------------------------------------------------------
    if epsilon_decay_episodes is None:
        epsilon_decay_episodes = int(num_episodes * 0.8)

    total_target = start_episode + num_episodes

    def _compute_epsilon(ep: int) -> float:
        """Linear interpolation from epsilon_start to epsilon_end."""
        episodes_into_decay = ep - start_episode
        if episodes_into_decay >= epsilon_decay_episodes:
            return epsilon_end
        progress = episodes_into_decay / max(epsilon_decay_episodes, 1)
        return epsilon_start + progress * (epsilon_end - epsilon_start)

    # ------------------------------------------------------------------
    # Graceful Ctrl+C handling (ported from C++ project)
    # ------------------------------------------------------------------
    stop_requested = False
    original_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        if stop_requested:
            # Second Ctrl+C: force exit immediately
            signal.signal(signal.SIGINT, original_handler)
            raise KeyboardInterrupt
        stop_requested = True
        print("\n[SIGINT] Finishing current episode and saving checkpoint...")

    signal.signal(signal.SIGINT, _sigint_handler)

    # ------------------------------------------------------------------
    # CSV logger
    # ------------------------------------------------------------------
    csv_path = run_dir / "train_history.csv"
    append_csv = resume and csv_path.exists()
    logger = CSVLogger(csv_path, append=append_csv)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    last_completed = start_episode

    window_return_sum = 0.0
    window_length_sum = 0
    window_success_count = 0
    window_collisions_sum = 0
    window_episodes_with_collision = 0
    window_episodes = 0

    try:
        for ep in range(start_episode + 1, start_episode + num_episodes + 1):
            ep_start = time.perf_counter()

            # ---- Run one episode ----
            obs, _info = env.reset()
            action = agent.epsilon_greedy_action(obs)
            episode_return = 0.0
            episode_length = 0
            episode_collisions = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                obs_next, reward, terminated, truncated, info = env.step(action)
                action_next = agent.epsilon_greedy_action(obs_next)
                agent.update(obs, action, reward, obs_next, action_next, terminated)
                obs = obs_next
                action = action_next
                episode_return += reward
                episode_length += 1
                episode_collisions += int(info.get("hit_obstacle", False))

            window_return_sum += episode_return
            window_length_sum += episode_length
            window_success_count += int(terminated)
            window_collisions_sum += episode_collisions
            if episode_collisions > 0:
                window_episodes_with_collision += 1
            window_episodes += 1

            # ---- Epsilon decay ----
            agent.epsilon = _compute_epsilon(ep)

            # ---- Log metrics ----
            ep_time = time.perf_counter() - ep_start
            logger.log({
                "episode": ep,
                "episode_return": round(episode_return, 4),
                "episode_length": episode_length,
                "success": int(terminated),
                "collisions": episode_collisions,
                "epsilon": round(agent.epsilon, 6),
                "episode_time_sec": round(ep_time, 6),
            })

            if ep % flush_every == 0:
                logger.flush()

            # ---- Print progress ----
            if ep % print_every == 0:
                elapsed = time.perf_counter() - t_start
                avg_return = window_return_sum / max(1, window_episodes)
                avg_len = window_length_sum / max(1, window_episodes)
                success_rate = (window_success_count / max(1, window_episodes)) * 100
                avg_coll = window_collisions_sum / max(1, window_episodes)
                coll_rate = (window_episodes_with_collision / max(1, window_episodes)) * 100

                print(
                    f"Ep {ep:>7d}/{total_target} | "
                    f"AvgReturn={avg_return:>8.2f} | "
                    f"AvgLen={avg_len:>6.1f} | "
                    f"AvgColl={avg_coll:>5.1f} | "
                    f"CollEp={coll_rate:>5.1f}% | "
                    f"Success={success_rate:>5.1f}% | "
                    f"ε={agent.epsilon:.4f} | "
                    f"Time={elapsed:.1f}s"
                )

                window_return_sum = 0.0
                window_length_sum = 0
                window_success_count = 0
                window_collisions_sum = 0
                window_episodes_with_collision = 0
                window_episodes = 0

            # ---- Periodic checkpoint ----
            if ep % checkpoint_every == 0:
                _save_checkpoint(checkpoint_dir, agent, ep)

            last_completed = ep

            # ---- Graceful stop ----
            if stop_requested:
                print(f"[SIGINT] Stopped at episode {ep}.")
                break

    finally:
        # Always save checkpoint on exit (normal or interrupted)
        _save_checkpoint(checkpoint_dir, agent, last_completed)
        logger.flush()
        logger.close()
        signal.signal(signal.SIGINT, original_handler)

    total_time = time.perf_counter() - t_start
    completed = last_completed - start_episode
    print(
        f"\nTraining finished: {completed} episodes in {total_time:.1f}s "
        f"({total_time / max(completed, 1):.4f} s/ep)"
    )

    return run_dir


def _save_checkpoint(
    checkpoint_dir: Path,
    agent: TabularAgent,
    episode: int,
) -> None:
    """Save agent state and training metadata to checkpoint directory."""
    # Save Q-table and agent metadata
    agent.save(checkpoint_dir)

    # Add training-specific metadata (episode count)
    meta_path = checkpoint_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata["last_completed_episode"] = episode
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
