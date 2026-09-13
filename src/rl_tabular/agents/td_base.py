"""Base class for Temporal-Difference (TD) agents."""

from __future__ import annotations

from rl_tabular.agents.base import TabularAgent


class TDAgent(TabularAgent):
    """
    Base class for Temporal-Difference agents.
    
    All TD agents need a learning rate (alpha) to perform updates.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize TD agent.

        Args:
            num_states: Size of the state space |S|.
            num_actions: Size of the action space |A|.
            alpha: Learning rate (step size).
            gamma: Discount factor.
            epsilon: Initial exploration rate for ε-greedy.
            seed: Random seed for reproducibility.
        """
        super().__init__(num_states, num_actions, gamma, epsilon, seed)
        self.alpha = alpha
