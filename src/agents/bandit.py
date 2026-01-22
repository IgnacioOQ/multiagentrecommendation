"""
Contextual Bandit Agents.

Implements:
- ContextualBanditAgent: General contextual bandit with ε-greedy, UCB, softmax exploration
- LinUCBAgent: Disjoint LinUCB with per-arm ridge regression and Sherman-Morrison updates
- RandomAgent: Random action baseline

Suitable for:
- Recommender systems with context-dependent preferences
- Smart Grid demand response (discrete action buckets)
- Server load balancing (routing decisions)
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

from .base import BaseAgent
from ..utils.math_ops import sherman_morrison_update


class ExplorationStrategy(Enum):
    """Exploration strategies for contextual bandits."""
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    SOFTMAX = "softmax"


class ContextualBanditAgent(BaseAgent):
    """General Contextual Bandit Agent with multiple exploration strategies.

    Maintains per-arm statistics and supports three exploration methods:
    - Epsilon-greedy: Random exploration with probability ε
    - UCB (Upper Confidence Bound): Optimism in the face of uncertainty
    - Softmax (Boltzmann): Probability proportional to exp(Q/τ)

    The agent learns a reward estimate for each (context, arm) pair.
    Context is discretized into bins for tabular estimation.

    Attributes:
        n_arms: Number of discrete actions/arms.
        n_context_bins: Number of bins for context discretization.
        exploration: Exploration strategy to use.
        epsilon: Exploration probability for ε-greedy.
        ucb_c: Exploration constant for UCB.
        temperature: Temperature for softmax exploration.
    """

    def __init__(
        self,
        n_arms: int,
        n_context_bins: int = 10,
        exploration: Literal["epsilon_greedy", "ucb", "softmax"] = "epsilon_greedy",
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        ucb_c: float = 2.0,
        temperature: float = 1.0,
        temperature_decay: float = 1.0,
        temperature_min: float = 0.1,
        optimistic_init: float = 0.0,
        name: str = "ContextualBandit",
    ):
        """Initialize the Contextual Bandit agent.

        Args:
            n_arms: Number of discrete actions/arms.
            n_context_bins: Number of bins for discretizing continuous context.
            exploration: Strategy - "epsilon_greedy", "ucb", or "softmax".
            epsilon: Initial exploration probability for ε-greedy.
            epsilon_decay: Multiplicative decay per step for epsilon.
            epsilon_min: Minimum epsilon value.
            ucb_c: Exploration constant for UCB (higher = more exploration).
            temperature: Initial temperature for softmax (higher = more exploration).
            temperature_decay: Multiplicative decay per step for temperature.
            temperature_min: Minimum temperature value.
            optimistic_init: Optimistic initialization for Q-values.
            name: Agent identifier.
        """
        super().__init__(name=name)

        self.n_arms = n_arms
        self.n_context_bins = n_context_bins
        self.exploration = ExplorationStrategy(exploration)

        # Epsilon-greedy parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self._initial_epsilon = epsilon

        # UCB parameters
        self.ucb_c = ucb_c

        # Softmax parameters
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min
        self._initial_temperature = temperature

        # Q-values: shape (n_context_bins, n_arms)
        self.Q = np.full((n_context_bins, n_arms), optimistic_init, dtype=np.float64)

        # Counts: shape (n_context_bins, n_arms)
        self.N = np.zeros((n_context_bins, n_arms), dtype=np.int64)

        # Total counts per context (for UCB)
        self.N_context = np.zeros(n_context_bins, dtype=np.int64)

        # Pending transition
        self._last_context_bin: Optional[int] = None
        self._last_action: Optional[int] = None
        self._pending_reward: float = 0.0

        # Statistics
        self.total_steps = 0

    def _discretize_context(self, state: np.ndarray) -> int:
        """Discretize continuous context into a bin index.

        Uses the first element of state or hash of full state vector.

        Args:
            state: Context vector.

        Returns:
            Bin index in [0, n_context_bins).
        """
        state = np.asarray(state, dtype=np.float64).flatten()

        if len(state) == 1:
            # Single-dimensional context: direct binning
            # Assume context is normalized to [0, 1]
            val = np.clip(state[0], 0.0, 1.0 - 1e-9)
            return int(val * self.n_context_bins)
        else:
            # Multi-dimensional: use hash-based binning
            # Normalize and hash
            state_normalized = (state - state.min()) / (state.max() - state.min() + 1e-9)
            hash_val = hash(tuple(np.round(state_normalized, decimals=2)))
            return abs(hash_val) % self.n_context_bins

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> int:
        """Select an action using the configured exploration strategy.

        Args:
            state: Context vector.
            explore: Whether to explore. If False, always exploit.

        Returns:
            Selected arm index.
        """
        context_bin = self._discretize_context(state)
        q_values = self.Q[context_bin]
        counts = self.N[context_bin]

        if not explore or not self.training:
            # Pure exploitation
            action = int(np.argmax(q_values))
        elif self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            action = self._select_epsilon_greedy(q_values)
        elif self.exploration == ExplorationStrategy.UCB:
            action = self._select_ucb(q_values, counts, context_bin)
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            action = self._select_softmax(q_values)
        else:
            raise ValueError(f"Unknown exploration strategy: {self.exploration}")

        # Store for update
        self._last_context_bin = context_bin
        self._last_action = action

        return action

    def _select_epsilon_greedy(self, q_values: np.ndarray) -> int:
        """Select action using epsilon-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        return int(np.argmax(q_values))

    def _select_ucb(
        self, q_values: np.ndarray, counts: np.ndarray, context_bin: int
    ) -> int:
        """Select action using Upper Confidence Bound."""
        total_count = self.N_context[context_bin]

        if total_count == 0:
            # No data yet, select randomly
            return np.random.randint(0, self.n_arms)

        # UCB formula: Q(a) + c * sqrt(ln(N) / N(a))
        ucb_values = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            if counts[a] == 0:
                # Unvisited arm gets infinite UCB (will be selected)
                ucb_values[a] = float('inf')
            else:
                exploration_bonus = self.ucb_c * np.sqrt(
                    np.log(total_count) / counts[a]
                )
                ucb_values[a] = q_values[a] + exploration_bonus

        return int(np.argmax(ucb_values))

    def _select_softmax(self, q_values: np.ndarray) -> int:
        """Select action using softmax (Boltzmann) exploration."""
        # Compute softmax probabilities: exp(Q/τ) / sum(exp(Q/τ))
        scaled = q_values / self.temperature

        # Numerical stability: subtract max
        scaled = scaled - np.max(scaled)
        exp_scaled = np.exp(scaled)
        probs = exp_scaled / np.sum(exp_scaled)

        return int(np.random.choice(self.n_arms, p=probs))

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition for learning."""
        self._pending_reward = reward

    def update(self) -> Dict[str, float]:
        """Update Q-values using incremental mean update.

        Returns:
            Training metrics.
        """
        if self._last_context_bin is None or self._last_action is None:
            return {}

        ctx = self._last_context_bin
        a = self._last_action
        r = self._pending_reward

        # Incremental mean update: Q = Q + (r - Q) / N
        self.N[ctx, a] += 1
        self.N_context[ctx] += 1
        n = self.N[ctx, a]

        old_q = self.Q[ctx, a]
        self.Q[ctx, a] = old_q + (r - old_q) / n

        # Decay exploration parameters
        if self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            self.temperature = max(
                self.temperature_min,
                self.temperature * self.temperature_decay
            )

        self.total_steps += 1

        # Clear pending
        self._last_context_bin = None
        self._last_action = None

        return {
            "context_bin": ctx,
            "arm_selected": a,
            "reward": r,
            "q_value": self.Q[ctx, a],
            "arm_count": n,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
        }

    def ready_to_train(self) -> bool:
        """Contextual bandit updates after every step."""
        return self._last_action is not None

    def get_q_table(self) -> np.ndarray:
        """Get the full Q-table.

        Returns:
            Q-values of shape (n_context_bins, n_arms).
        """
        return self.Q.copy()

    def get_policy(self) -> np.ndarray:
        """Get the greedy policy.

        Returns:
            Array of shape (n_context_bins,) with best action per context.
        """
        return np.argmax(self.Q, axis=1)

    def save(self, path: str) -> None:
        """Save agent parameters."""
        np.savez(
            path,
            Q=self.Q,
            N=self.N,
            N_context=self.N_context,
            epsilon=self.epsilon,
            temperature=self.temperature,
            total_steps=self.total_steps,
            exploration=self.exploration.value,
        )

    def load(self, path: str) -> None:
        """Load agent parameters."""
        data = np.load(path)
        self.Q = data["Q"]
        self.N = data["N"]
        self.N_context = data["N_context"]
        self.epsilon = float(data["epsilon"])
        self.temperature = float(data["temperature"])
        self.total_steps = int(data["total_steps"])
        self.exploration = ExplorationStrategy(str(data["exploration"]))

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            **super().get_config(),
            "n_arms": self.n_arms,
            "n_context_bins": self.n_context_bins,
            "exploration": self.exploration.value,
            "epsilon": self._initial_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "ucb_c": self.ucb_c,
            "temperature": self._initial_temperature,
            "temperature_decay": self.temperature_decay,
            "temperature_min": self.temperature_min,
        }

    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.Q = np.zeros((self.n_context_bins, self.n_arms), dtype=np.float64)
        self.N = np.zeros((self.n_context_bins, self.n_arms), dtype=np.int64)
        self.N_context = np.zeros(self.n_context_bins, dtype=np.int64)
        self.epsilon = self._initial_epsilon
        self.temperature = self._initial_temperature
        self.total_steps = 0
        self._last_context_bin = None
        self._last_action = None


class RandomAgent(BaseAgent):
    """Random action selection baseline agent.

    Selects actions uniformly at random, ignoring the state.
    Useful as a baseline for comparing against learned policies.

    Attributes:
        n_actions: Number of discrete actions available.
    """

    def __init__(self, n_actions: int, name: str = "Random"):
        """Initialize the Random agent.

        Args:
            n_actions: Number of discrete actions available.
            name: Agent identifier.
        """
        super().__init__(name=name)
        self.n_actions = n_actions

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> int:
        """Select a random action.

        Args:
            state: Current state (ignored).
            explore: Whether to explore (ignored - always random).

        Returns:
            Randomly selected action index.
        """
        return np.random.randint(0, self.n_actions)

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition (no-op for random agent)."""
        pass

    def update(self) -> Dict[str, float]:
        """Update the model (no-op for random agent)."""
        return {}

    def ready_to_train(self) -> bool:
        """Random agent never trains."""
        return False

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            **super().get_config(),
            "n_actions": self.n_actions,
        }


class LinUCBAgent(BaseAgent):
    """Disjoint LinUCB Contextual Bandit Agent.

    Maintains a separate ridge regression model for each arm (action).
    Uses Upper Confidence Bound (UCB) for exploration-exploitation balance.

    The expected reward for arm a given context x is modeled as:
        E[r | x, a] = x^T θ_a

    At each step, the agent:
    1. Computes UCB for each arm: p_a = θ̂_a^T x + α √(x^T A_a^{-1} x)
    2. Selects the arm with highest UCB
    3. Updates the model for the selected arm after receiving reward

    Attributes:
        n_arms: Number of discrete actions.
        context_dim: Dimension of context/state features.
        alpha: Exploration parameter (higher = more exploration).
        regularization: Ridge regression regularization (λ).
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        regularization: float = 1.0,
        name: str = "LinUCB",
    ):
        """Initialize the LinUCB agent.

        Args:
            n_arms: Number of discrete actions/arms.
            context_dim: Dimension of the context vector.
            alpha: UCB exploration parameter.
            regularization: Ridge regression regularization λ.
            name: Agent identifier.
        """
        super().__init__(name=name)
        
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.regularization = regularization
        
        # Per-arm parameters
        # A_inv[a] = (A_a)^{-1} where A_a = X_a^T X_a + λI
        # b[a] = X_a^T r_a
        self.A_inv = np.array([
            np.eye(context_dim) / regularization for _ in range(n_arms)
        ])
        self.b = np.zeros((n_arms, context_dim))
        
        # Cached theta estimates
        self._theta = np.zeros((n_arms, context_dim))
        
        # Last action taken (for update)
        self._last_action: Optional[int] = None
        self._last_context: Optional[np.ndarray] = None
        
        # Statistics
        self.arm_counts = np.zeros(n_arms, dtype=np.int64)
        self.total_steps = 0

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> int:
        """Select an action using UCB criterion.

        Args:
            state: Context vector (1D array of shape (context_dim,)).
            explore: Whether to use UCB exploration. If False, uses greedy.

        Returns:
            Selected arm index.
        """
        x = np.asarray(state, dtype=np.float64).flatten()
        
        if len(x) != self.context_dim:
            raise ValueError(
                f"State dimension {len(x)} doesn't match context_dim {self.context_dim}"
            )
        
        # Compute UCB for each arm
        ucb_scores = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            # θ̂_a = A_a^{-1} b_a
            theta_a = self.A_inv[a] @ self.b[a]
            self._theta[a] = theta_a
            
            # Mean estimate: θ̂_a^T x
            mean = np.dot(theta_a, x)
            
            if explore and self.training:
                # UCB bonus: α √(x^T A_a^{-1} x)
                x_A_inv_x = x @ self.A_inv[a] @ x
                ucb_bonus = self.alpha * np.sqrt(max(0.0, x_A_inv_x))
                ucb_scores[a] = mean + ucb_bonus
            else:
                ucb_scores[a] = mean
        
        # Select arm with highest UCB
        action = int(np.argmax(ucb_scores))
        
        # Store for update
        self._last_action = action
        self._last_context = x.copy()
        
        return action

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition (updates happen immediately in update()).

        For LinUCB, we store the context and reward for the selected arm
        to update the model in the update() call.
        """
        # Context and action already stored in select_action
        self._pending_reward = reward

    def update(self) -> Dict[str, float]:
        """Update the model for the last selected arm.

        Uses Sherman-Morrison formula for O(d²) inverse update.

        Returns:
            Training metrics (arm selected, reward received).
        """
        if self._last_action is None or self._last_context is None:
            return {}
        
        a = self._last_action
        x = self._last_context
        r = self._pending_reward
        
        # Update A_inv using Sherman-Morrison
        # A_new = A_old + x x^T
        # A_new^{-1} = A_old^{-1} - (A_old^{-1} x x^T A_old^{-1}) / (1 + x^T A_old^{-1} x)
        self.A_inv[a] = sherman_morrison_update(self.A_inv[a], x)
        
        # Update b: b_new = b_old + r * x
        self.b[a] += r * x
        
        # Update statistics
        self.arm_counts[a] += 1
        self.total_steps += 1
        
        # Clear pending
        self._last_action = None
        self._last_context = None
        
        return {
            "arm_selected": a,
            "reward": r,
            "arm_count": self.arm_counts[a],
        }

    def ready_to_train(self) -> bool:
        """LinUCB updates after every step."""
        return self._last_action is not None

    def get_arm_estimates(self, context: np.ndarray) -> np.ndarray:
        """Get estimated rewards for all arms given a context.

        Args:
            context: Context vector.

        Returns:
            Array of shape (n_arms,) with estimated rewards.
        """
        x = np.asarray(context, dtype=np.float64).flatten()
        estimates = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            theta_a = self.A_inv[a] @ self.b[a]
            estimates[a] = np.dot(theta_a, x)
        
        return estimates

    def get_arm_uncertainties(self, context: np.ndarray) -> np.ndarray:
        """Get uncertainty estimates for all arms given a context.

        Args:
            context: Context vector.

        Returns:
            Array of shape (n_arms,) with uncertainty (√(x^T A^{-1} x)).
        """
        x = np.asarray(context, dtype=np.float64).flatten()
        uncertainties = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            x_A_inv_x = x @ self.A_inv[a] @ x
            uncertainties[a] = np.sqrt(max(0.0, x_A_inv_x))
        
        return uncertainties

    def save(self, path: str) -> None:
        """Save agent parameters."""
        np.savez(
            path,
            A_inv=self.A_inv,
            b=self.b,
            arm_counts=self.arm_counts,
            total_steps=self.total_steps,
            alpha=self.alpha,
            regularization=self.regularization,
        )

    def load(self, path: str) -> None:
        """Load agent parameters."""
        data = np.load(path)
        self.A_inv = data["A_inv"]
        self.b = data["b"]
        self.arm_counts = data["arm_counts"]
        self.total_steps = int(data["total_steps"])
        self.alpha = float(data["alpha"])
        self.regularization = float(data["regularization"])

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            **super().get_config(),
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "regularization": self.regularization,
        }

    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.A_inv = np.array([
            np.eye(self.context_dim) / self.regularization 
            for _ in range(self.n_arms)
        ])
        self.b = np.zeros((self.n_arms, self.context_dim))
        self._theta = np.zeros((self.n_arms, self.context_dim))
        self.arm_counts = np.zeros(self.n_arms, dtype=np.int64)
        self.total_steps = 0
        self._last_action = None
        self._last_context = None
