"""SARSA agent (on-policy TD control)."""

from __future__ import annotations

from rl_tabular.agents.td_base import TDAgent


class SarsaAgent(TDAgent):
    """
    SARSA agent — on-policy TD(0).

    TODO (Luiz Miguel): Implement the update rule for SARSA.
    Remember that unlike Q-Learning, SARSA is on-policy, which means
    it needs to know the exact action 'a_next' that the agent chose
    for the next state 's_next'.
    """

    def update(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        a_next: int,
        terminated: bool,
    ) -> None:
        """
        Perform one SARSA update.

        Args:
            s: Current state.
            a: Action taken.
            r: Reward received.
            s_next: Next state after taking action a in state s.
            a_next: Next action chosen by the behavior policy in state s_next.
            terminated: True if the episode ended (goal reached).
                        When True, the bootstrap term is zero.
        """
        if terminated:
            target = r
        else:
            target = r + self.gamma * self.Q[s_next, a_next]

        td_error = target - self.Q[s, a]
        self.Q[s, a] = self.Q[s, a] + self.alpha * td_error
