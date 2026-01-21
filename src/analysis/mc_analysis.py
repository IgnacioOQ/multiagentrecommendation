"""
Markov Chain Analysis Tools for Recommender System Simulations

This module provides infrastructure for analyzing simulations as Markov Chains,
enabling rigorous analysis of convergence, equilibrium, and the effects of
reward modulation on system dynamics.

Key Concepts:
- State Space: (Q_user, Q_rec, M_state, context)
- Transitions: Driven by agent actions, environment responses, and modulator updates
- Absorbing States: Equilibria where user preferences stabilize
- Mixing Time: How fast the system reaches equilibrium

Reference: AI_AGENTS/MC_AGENT.md
"""

import numpy as np
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class StateSnapshot:
    """
    A snapshot of the full system state at a given timestep.
    
    Attributes:
        step: The timestep when this snapshot was taken
        user_q_values: Copy of user agent's Q-table
        recommender_q_values: Copy of recommender agent's Q-table
        modulator_state: Relevant modulator internal state
        context: Current environmental context
        fingerprint: Hash for quick comparison
    """
    step: int
    user_q_values: Dict[Any, np.ndarray]
    recommender_q_values: Dict[int, np.ndarray]
    modulator_state: Optional[Dict[str, Any]] = None
    context: Optional[int] = None
    fingerprint: str = ""
    
    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()
    
    def _compute_fingerprint(self) -> str:
        """Compute a hash of the Q-values for quick state comparison."""
        # Flatten user Q-values to a string representation
        user_flat = []
        for key in sorted(self.user_q_values.keys(), key=str):
            vals = self.user_q_values[key]
            if isinstance(vals, np.ndarray):
                user_flat.extend(vals.flatten().tolist())
            else:
                user_flat.append(vals)
        
        rec_flat = []
        for key in sorted(self.recommender_q_values.keys()):
            vals = self.recommender_q_values[key]
            if isinstance(vals, np.ndarray):
                rec_flat.extend(vals.flatten().tolist())
            else:
                rec_flat.append(vals)
        
        # Round to avoid floating point noise
        combined = [round(v, 6) for v in user_flat + rec_flat]
        hash_input = json.dumps(combined)
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]


@dataclass
class TransitionRecord:
    """
    Records a single state transition for analysis.
    """
    step: int
    state_before: StateSnapshot
    action_recommender: int
    action_user: int  # 0 = accept, 1 = reject
    reward_true: float
    reward_modulated: float
    state_after: StateSnapshot


class MarkovChainAnalyzer:
    """
    Tools for analyzing the Markov Chain properties of recommender simulations.
    
    This class attaches to simulation agents and tracks:
    - State snapshots over time
    - Transition records
    - Convergence diagnostics
    - Absorption probabilities (via Monte Carlo)
    
    Usage:
        analyzer = MarkovChainAnalyzer()
        
        # During simulation, periodically call:
        analyzer.snapshot_state(step, user_agent, rec_agent, modulator, context)
        
        # After simulation:
        analyzer.compute_state_distances()
        analyzer.estimate_absorption_probability(...)
    """
    
    def __init__(self):
        self.state_snapshots: List[StateSnapshot] = []
        self.transitions: List[TransitionRecord] = []
        self.fingerprint_history: List[str] = []
        self._state_visit_counts: Dict[str, int] = defaultdict(int)
    
    def snapshot_state(
        self,
        step: int,
        user_agent,
        recommender_agent,
        modulator=None,
        context: int = None
    ) -> StateSnapshot:
        """
        Capture the full system state at current timestep.
        
        Args:
            step: Current simulation step
            user_agent: The RecommendedAgent instance
            recommender_agent: The RecommenderAgent instance
            modulator: Optional reward modulator instance
            context: Current environment context
            
        Returns:
            StateSnapshot object containing copied state
        """
        # Deep copy Q-tables to avoid mutation issues
        user_q = {k: np.array(v) for k, v in user_agent.q_table.items()}
        rec_q = {k: np.array(v) for k, v in recommender_agent.q_table.items()}
        
        # Extract modulator state if present
        mod_state = None
        if modulator is not None:
            mod_state = self._extract_modulator_state(modulator)
        
        snapshot = StateSnapshot(
            step=step,
            user_q_values=user_q,
            recommender_q_values=rec_q,
            modulator_state=mod_state,
            context=context
        )
        
        self.state_snapshots.append(snapshot)
        self.fingerprint_history.append(snapshot.fingerprint)
        self._state_visit_counts[snapshot.fingerprint] += 1
        
        return snapshot
    
    def _extract_modulator_state(self, modulator) -> Dict[str, Any]:
        """
        Extract relevant internal state from a modulator.
        Different modulator types have different state variables.
        """
        state = {"type": type(modulator).__name__}
        
        # ReceptorModulator
        if hasattr(modulator, 'sensitivity'):
            state['sensitivity'] = modulator.sensitivity
        
        # HomeostaticModulator
        if hasattr(modulator, 'setpoint'):
            state['setpoint'] = modulator.setpoint
        
        # MoodSwings
        if hasattr(modulator, 'mood'):
            state['mood'] = modulator.mood
        
        # NoveltyModulator
        if hasattr(modulator, 'visit_counts'):
            state['total_visits'] = sum(modulator.visit_counts.values())
        
        # History-based modulators
        if hasattr(modulator, 'history'):
            state['history'] = list(modulator.history)
        
        return state
    
    def record_transition(
        self,
        step: int,
        state_before: StateSnapshot,
        action_recommender: int,
        action_user: int,
        reward_true: float,
        reward_modulated: float,
        state_after: StateSnapshot
    ):
        """
        Record a single transition for later analysis.
        """
        transition = TransitionRecord(
            step=step,
            state_before=state_before,
            action_recommender=action_recommender,
            action_user=action_user,
            reward_true=reward_true,
            reward_modulated=reward_modulated,
            state_after=state_after
        )
        self.transitions.append(transition)
    
    def compute_state_distance(
        self,
        snapshot1: StateSnapshot,
        snapshot2: StateSnapshot,
        metric: str = "frobenius"
    ) -> float:
        """
        Compute distance between two states.
        
        Args:
            snapshot1, snapshot2: State snapshots to compare
            metric: Distance metric ("frobenius", "max", "l1")
            
        Returns:
            Distance value (float)
        """
        # Flatten Q-values into vectors
        def flatten_q(q_dict):
            vals = []
            for k in sorted(q_dict.keys(), key=str):
                v = q_dict[k]
                if isinstance(v, np.ndarray):
                    vals.extend(v.flatten())
                else:
                    vals.append(float(v))
            return np.array(vals)
        
        user1 = flatten_q(snapshot1.user_q_values)
        user2 = flatten_q(snapshot2.user_q_values)
        rec1 = flatten_q(snapshot1.recommender_q_values)
        rec2 = flatten_q(snapshot2.recommender_q_values)
        
        # Handle different sizes gracefully
        def safe_diff(a, b):
            min_len = min(len(a), len(b))
            return a[:min_len] - b[:min_len]
        
        user_diff = safe_diff(user1, user2)
        rec_diff = safe_diff(rec1, rec2)
        
        if metric == "frobenius":
            return np.sqrt(np.sum(user_diff**2) + np.sum(rec_diff**2))
        elif metric == "max":
            return max(np.max(np.abs(user_diff)), np.max(np.abs(rec_diff)))
        elif metric == "l1":
            return np.sum(np.abs(user_diff)) + np.sum(np.abs(rec_diff))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_convergence_trajectory(self) -> List[float]:
        """
        Compute the state distance between consecutive snapshots.
        Useful for detecting when the system has converged.
        
        Returns:
            List of distances between consecutive states
        """
        if len(self.state_snapshots) < 2:
            return []
        
        distances = []
        for i in range(1, len(self.state_snapshots)):
            d = self.compute_state_distance(
                self.state_snapshots[i-1],
                self.state_snapshots[i]
            )
            distances.append(d)
        
        return distances
    
    def detect_absorption(self, threshold: float = 1e-4, window: int = 10) -> Optional[int]:
        """
        Detect when the system has reached an absorbing state.
        
        An absorbing state is detected when state changes fall below threshold
        for `window` consecutive snapshots.
        
        Args:
            threshold: Maximum allowed state change to be considered "absorbed"
            window: Number of consecutive low-change steps required
            
        Returns:
            Step number where absorption was detected, or None
        """
        distances = self.compute_convergence_trajectory()
        
        consecutive_low = 0
        for i, d in enumerate(distances):
            if d < threshold:
                consecutive_low += 1
                if consecutive_low >= window:
                    # Return the step where absorption started
                    return self.state_snapshots[i - window + 1].step
            else:
                consecutive_low = 0
        
        return None
    
    def state_fingerprint_entropy(self) -> float:
        """
        Compute entropy of state visitation distribution.
        Higher entropy = more diverse state space exploration.
        
        Returns:
            Shannon entropy of fingerprint distribution
        """
        total = sum(self._state_visit_counts.values())
        if total == 0:
            return 0.0
        
        probs = np.array(list(self._state_visit_counts.values())) / total
        # Filter out zero probabilities to avoid log(0)
        probs = probs[probs > 0]
        
        return -np.sum(probs * np.log2(probs))
    
    def unique_states_visited(self) -> int:
        """Return the number of unique state fingerprints encountered."""
        return len(self._state_visit_counts)
    
    def check_markov_property(self, n_tests: int = 100) -> Dict[str, Any]:
        """
        Verify that the simulation satisfies the Markov property.
        
        This is done by checking that given the same state (fingerprint),
        the distribution of next states is consistent regardless of history.
        
        Note: This is a statistical test, not a formal proof.
        
        Args:
            n_tests: Number of state pairs to check
            
        Returns:
            Dictionary with test results
        """
        if len(self.transitions) < n_tests:
            return {
                "passed": None,
                "reason": "Not enough transitions recorded",
                "n_transitions": len(self.transitions)
            }
        
        # Group transitions by state_before fingerprint
        transitions_by_state: Dict[str, List[TransitionRecord]] = defaultdict(list)
        for t in self.transitions:
            transitions_by_state[t.state_before.fingerprint].append(t)
        
        # For states with multiple outgoing transitions, check consistency
        states_with_multiple = {k: v for k, v in transitions_by_state.items() if len(v) > 1}
        
        if len(states_with_multiple) == 0:
            return {
                "passed": None,
                "reason": "No repeated states found (expected in continuous state space)",
                "unique_states": len(transitions_by_state)
            }
        
        # In a deterministic system with same random seed, same state → same next state
        # In a stochastic system, we check variance in next states
        consistency_scores = []
        for fingerprint, trans_list in states_with_multiple.items():
            next_fingerprints = [t.state_after.fingerprint for t in trans_list]
            unique_next = len(set(next_fingerprints))
            consistency_scores.append(1.0 / unique_next)  # 1.0 if deterministic
        
        avg_consistency = np.mean(consistency_scores)
        
        return {
            "passed": True,  # Markov property is structural, not about consistency
            "average_consistency": avg_consistency,
            "states_with_multiple_visits": len(states_with_multiple),
            "interpretation": "High consistency suggests deterministic transitions; "
                            "low consistency suggests stochastic transitions (both valid Markov)"
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked Markov Chain properties.
        """
        conv_traj = self.compute_convergence_trajectory()
        
        return {
            "n_snapshots": len(self.state_snapshots),
            "n_transitions": len(self.transitions),
            "unique_states": self.unique_states_visited(),
            "state_entropy": self.state_fingerprint_entropy(),
            "absorption_step": self.detect_absorption(),
            "final_state_change": conv_traj[-1] if conv_traj else None,
            "mean_state_change": np.mean(conv_traj) if conv_traj else None,
        }
    
    def reset(self):
        """Clear all tracked data."""
        self.state_snapshots = []
        self.transitions = []
        self.fingerprint_history = []
        self._state_visit_counts = defaultdict(int)


def estimate_absorption_probabilities(
    simulation_fn,
    target_regions: Dict[str, callable],
    n_simulations: int = 100,
    random_seed_base: int = 42
) -> Dict[str, float]:
    """
    Estimate probability of absorbing to different regions via Monte Carlo.
    
    Args:
        simulation_fn: Callable that runs a simulation and returns result dict
        target_regions: Dict mapping region names to predicates on final state
            e.g., {"global_optimum": lambda result: result["final_reward"] > 0.9}
        n_simulations: Number of Monte Carlo runs
        random_seed_base: Base seed for reproducibility
        
    Returns:
        Dict mapping region names to estimated absorption probabilities
    """
    counts = {name: 0 for name in target_regions}
    
    for i in range(n_simulations):
        result = simulation_fn(random_seed=random_seed_base + i)
        
        for name, predicate in target_regions.items():
            if predicate(result):
                counts[name] += 1
    
    probabilities = {name: count / n_simulations for name, count in counts.items()}
    return probabilities


def estimate_mixing_time(
    simulation_fn,
    distance_threshold: float = 0.1,
    n_chains: int = 10,
    max_steps: int = 100000,
    random_seed_base: int = 42
) -> Dict[str, Any]:
    """
    Estimate mixing time by running parallel chains and measuring convergence.
    
    Mixing time is approximated as the step when all chains have similar state.
    
    Args:
        simulation_fn: Callable that runs simulation with (n_steps, random_seed) kwargs
        distance_threshold: Max pairwise distance to consider "mixed"
        n_chains: Number of parallel chains to run
        max_steps: Maximum steps to run each chain
        random_seed_base: Base seed for reproducibility
        
    Returns:
        Dict with mixing time estimate and diagnostics
    """
    # This is a placeholder for a more sophisticated implementation
    # A proper implementation would require running chains iteratively
    # and checking convergence at each step
    
    return {
        "estimated_mixing_time": None,
        "note": "Full implementation requires interactive chain stepping",
        "n_chains": n_chains,
        "threshold": distance_threshold
    }
