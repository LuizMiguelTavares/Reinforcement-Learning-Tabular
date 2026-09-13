"""Q-Learning agent (off-policy TD control)."""

from __future__ import annotations

from rl_tabular.agents.td_base import TDAgent


class QLearningAgent(TDAgent):
    """
    Q-Learning agent — off-policy TD(0).

    Update rule (Sutton & Barto, Section 6.5):

        Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') - Q(s, a) ]

    The key insight is that Q-Learning uses the MAX over next-state actions
    in the target, regardless of which action the agent actually takes next.
    This makes it off-policy: the behavior policy (ε-greedy) can explore
    freely while learning the optimal Q* directly.
    """

    def update(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        terminated: bool,
    ) -> None:
        """
        Perform one Q-Learning update.

        Args:
            s: Current state.
            a: Action taken.
            r: Reward received.
            s_next: Next state after taking action a in state s.
            terminated: True if the episode ended (goal reached).
                        When True, the bootstrap term is zero.
        """
        # TD target: r + γ max_a' Q(s', a')
        # If terminated, there is no future reward → target = r
        if terminated:
            target = r
        else:
            target = r + self.gamma * self.Q[s_next].max()

        # TD error: δ = target - Q(s, a)
        td_error = target - self.Q[s, a]

        # Update: Q(s, a) ← Q(s, a) + α δ
        self.Q[s, a] += self.alpha * td_error
