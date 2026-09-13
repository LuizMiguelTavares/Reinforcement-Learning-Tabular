"""Expected SARSA agent (TD control)."""

from __future__ import annotations

from rl_tabular.agents.td_base import TDAgent


class ExpectedSarsaAgent(TDAgent):
    """
    Expected SARSA agent.

    TODO (Luiz Miguel): Implement the update rule for Expected SARSA.
    Unlike Q-Learning (which uses the max Q-value of the next state) and
    SARSA (which uses the Q-value of the exact next action chosen), Expected
    SARSA uses the *expected* value of the next state, considering the probability
    of taking each action under the current policy (e.g., epsilon-greedy).
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
        Perform one Expected SARSA update.

        Args:
            s: Current state.
            a: Action taken.
            r: Reward received.
            s_next: Next state after taking action a in state s.
            terminated: True if the episode ended (goal reached).
                        When True, the bootstrap term is zero.
        """
        # TODO: Implement the update rule here!
        # Hint: You'll need to compute the probability of taking each action in s_next
        # given your current self.epsilon value.
        pass
