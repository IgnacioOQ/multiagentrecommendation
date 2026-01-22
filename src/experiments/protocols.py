"""
Experiment protocols for the Multi-Agent Recommendation System.

Defines the three main protocols from TODOS.md:
- Protocol A: Stationary Baseline
- Protocol B: Modulated Learning
- Protocol C: Non-Stationary Environment

Each protocol validates configurations and computes protocol-specific metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any, Type, Optional

import numpy as np

from .config import ExperimentConfig, Protocol, ModulatorType


@dataclass
class ProtocolResult:
    """Results from running an experiment protocol."""
    protocol: Protocol
    metrics: Dict[str, float]
    simulation_results: Dict[str, Any]
    mc_analysis: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = None


class BaseProtocol:
    """Base class for experiment protocols."""

    protocol_type: Protocol = None

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def validate_config(self) -> None:
        """Validate that config is appropriate for this protocol."""
        pass

    def get_modulator_class(self) -> Optional[Type]:
        """Return the modulator class to use, or None."""
        from src.reward_modulators import (
            ReceptorModulator,
            NoveltyModulator,
            HomeostaticModulator,
            TD_HomeostaticModulator,
            MoodSwings,
        )

        mapping = {
            ModulatorType.NONE: None,
            ModulatorType.RECEPTOR: ReceptorModulator,
            ModulatorType.NOVELTY: NoveltyModulator,
            ModulatorType.HOMEOSTATIC: HomeostaticModulator,
            ModulatorType.TD_HOMEOSTATIC: TD_HomeostaticModulator,
            ModulatorType.MOOD_SWINGS: MoodSwings,
        }
        return mapping.get(self.config.modulator.modulator_type)

    def create_modulator_kwargs(self) -> Dict[str, Any]:
        """Create kwargs for modulator instantiation."""
        mc = self.config.modulator

        if mc.modulator_type == ModulatorType.RECEPTOR:
            return {
                "alpha": mc.alpha,
                "beta": mc.beta,
                "min_sensitivity": mc.min_sensitivity,
                "max_sensitivity": mc.max_sensitivity,
                "desensitization_threshold": mc.desensitization_threshold,
            }
        elif mc.modulator_type == ModulatorType.NOVELTY:
            return {"eta": mc.eta}
        elif mc.modulator_type in (
            ModulatorType.HOMEOSTATIC,
            ModulatorType.TD_HOMEOSTATIC
        ):
            return {
                "setpoint": mc.setpoint,
                "lag": mc.lag,
                "n_bins": mc.n_bins,
                "reward_range": mc.reward_range,
            }
        elif mc.modulator_type == ModulatorType.MOOD_SWINGS:
            return {}  # MoodSwings has n_moods default
        else:
            return {}

    def compute_metrics(self, sim_results: Dict[str, Any]) -> Dict[str, float]:
        """Compute protocol-specific metrics from simulation results."""
        rec_rewards = np.array(sim_results["recommender_rewards"])
        user_rewards = np.array(sim_results["recommended_rewards"])
        accept_history = sim_results["accept_history"]
        accepts = np.array([a[3] for a in accept_history])

        # Use last 10% of simulation for final metrics
        n_final = max(1, len(rec_rewards) // 10)

        return {
            "mean_recommender_reward": float(np.mean(rec_rewards)),
            "mean_user_reward": float(np.mean(user_rewards)),
            "final_recommender_reward": float(np.mean(rec_rewards[-n_final:])),
            "final_user_reward": float(np.mean(user_rewards[-n_final:])),
            "acceptance_rate": float(np.mean(accepts)),
            "final_acceptance_rate": float(np.mean(accepts[-n_final:])),
            "total_user_reward": float(np.sum(user_rewards)),
            "total_recommender_reward": float(np.sum(rec_rewards)),
        }


class ProtocolA_Stationary(BaseProtocol):
    """
    Protocol A: Stationary Baseline

    - No modulation (modulator=None)
    - Fixed environment (stationarity=True)
    - Measures: convergence time, Q-landscape accuracy
    """

    protocol_type = Protocol.A_STATIONARY

    def validate_config(self) -> None:
        if self.config.modulator.modulator_type != ModulatorType.NONE:
            raise ValueError(
                "Protocol A (Stationary Baseline) requires no modulation. "
                f"Got modulator_type={self.config.modulator.modulator_type}"
            )
        if not self.config.environment.stationarity:
            raise ValueError(
                "Protocol A (Stationary Baseline) requires stationary environment. "
                "Got stationarity=False"
            )

    def compute_metrics(self, sim_results: Dict[str, Any]) -> Dict[str, float]:
        base_metrics = super().compute_metrics(sim_results)

        # Add Protocol A specific metrics
        env_landscape = sim_results["environment_state_space"]
        final_qvalues = sim_results["final_user_qvalues"]

        # Q-landscape accuracy: correlation with true environment
        env_flat = env_landscape.flatten()
        q_flat = final_qvalues.flatten()

        # Handle NaN in Q-values (unexplored states)
        valid_mask = ~np.isnan(q_flat)
        if valid_mask.sum() > 1:
            correlation = np.corrcoef(
                env_flat[valid_mask],
                q_flat[valid_mask]
            )[0, 1]
        else:
            correlation = 0.0

        base_metrics["q_landscape_correlation"] = float(correlation)

        # Compute MSE between learned and true (for explored states)
        if valid_mask.sum() > 0:
            # Normalize both to [0, 1] for fair comparison
            env_norm = (env_flat - env_flat.min()) / (env_flat.max() - env_flat.min() + 1e-8)
            q_norm = (q_flat - np.nanmin(q_flat)) / (np.nanmax(q_flat) - np.nanmin(q_flat) + 1e-8)
            mse = np.mean((env_norm[valid_mask] - q_norm[valid_mask]) ** 2)
            base_metrics["q_landscape_mse"] = float(mse)
        else:
            base_metrics["q_landscape_mse"] = float("nan")

        # Fraction of state space explored
        total_states = env_landscape.size
        explored_states = valid_mask.sum()
        base_metrics["exploration_coverage"] = float(explored_states / total_states)

        return base_metrics


class ProtocolB_Modulated(BaseProtocol):
    """
    Protocol B: Modulated Learning

    - Apply reward modulators (ReceptorModulator, NoveltyModulator, etc.)
    - Fixed environment (stationarity=True)
    - Measures: how modulation affects learned preferences, lock-in rates
    """

    protocol_type = Protocol.B_MODULATED

    def validate_config(self) -> None:
        if self.config.modulator.modulator_type == ModulatorType.NONE:
            raise ValueError(
                "Protocol B (Modulated Learning) requires a modulator. "
                "Got modulator_type=NONE"
            )

    def compute_metrics(self, sim_results: Dict[str, Any]) -> Dict[str, float]:
        base_metrics = super().compute_metrics(sim_results)

        # Add modulation-specific metrics
        if "original_modulated_differences" in sim_results:
            diffs = np.array(sim_results["original_modulated_differences"])
            base_metrics["mean_modulation_magnitude"] = float(np.mean(np.abs(diffs)))
            base_metrics["modulation_variance"] = float(np.var(diffs))
            base_metrics["max_modulation"] = float(np.max(np.abs(diffs)))

        # Lock-in detection: Did the agent converge to local vs global optimum?
        final_qvalues = sim_results["final_user_qvalues"]
        env_landscape = sim_results["environment_state_space"]

        # Find true global max
        global_max_pos = np.unravel_index(
            np.argmax(env_landscape),
            env_landscape.shape
        )

        # Find learned max (ignoring NaN)
        q_masked = np.where(np.isnan(final_qvalues), -np.inf, final_qvalues)
        q_global_max_pos = np.unravel_index(
            np.argmax(q_masked),
            final_qvalues.shape
        )

        # Distance between learned max and true global max
        max_distance = np.sqrt(
            (global_max_pos[0] - q_global_max_pos[0]) ** 2 +
            (global_max_pos[1] - q_global_max_pos[1]) ** 2
        )
        base_metrics["optimum_distance"] = float(max_distance)

        # Check if converged to global (within 3 cells)
        base_metrics["converged_to_global"] = int(max_distance < 3)

        # Value at learned max vs true global max
        learned_max_value = env_landscape[q_global_max_pos]
        true_max_value = env_landscape[global_max_pos]
        base_metrics["suboptimality_gap"] = float(true_max_value - learned_max_value)
        base_metrics["relative_suboptimality"] = float(
            (true_max_value - learned_max_value) / (true_max_value + 1e-8)
        )

        return base_metrics


class ProtocolC_NonStationary(BaseProtocol):
    """
    Protocol C: Non-Stationary Environment

    - Environment shifts periodically (stationarity=False)
    - Measures: adaptation rate, tracking error
    """

    protocol_type = Protocol.C_NON_STATIONARY

    def validate_config(self) -> None:
        if self.config.environment.stationarity:
            raise ValueError(
                "Protocol C (Non-Stationary) requires non-stationary environment. "
                "Got stationarity=True"
            )

    def compute_metrics(self, sim_results: Dict[str, Any]) -> Dict[str, float]:
        base_metrics = super().compute_metrics(sim_results)

        user_rewards = np.array(sim_results["recommended_rewards"])

        # Compute variance in rolling windows to detect adaptation lag
        window_size = max(1, len(user_rewards) // 100)
        n_windows = len(user_rewards) // window_size

        if n_windows > 1:
            window_means = [
                np.mean(user_rewards[i * window_size:(i + 1) * window_size])
                for i in range(n_windows)
            ]
            base_metrics["reward_window_variance"] = float(np.var(window_means))

            # Trend: positive = improving, negative = declining
            x = np.arange(n_windows)
            slope = np.polyfit(x, window_means, 1)[0]
            base_metrics["reward_trend"] = float(slope)

            # Adaptation metric: how quickly rewards recover after each window
            # (Higher variance in early windows vs later suggests adaptation)
            mid_point = n_windows // 2
            early_var = np.var(window_means[:mid_point]) if mid_point > 1 else 0
            late_var = np.var(window_means[mid_point:]) if mid_point > 1 else 0
            base_metrics["adaptation_ratio"] = float(
                early_var / (late_var + 1e-8)
            )
        else:
            base_metrics["reward_window_variance"] = 0.0
            base_metrics["reward_trend"] = 0.0
            base_metrics["adaptation_ratio"] = 1.0

        # Cumulative regret analysis
        # In non-stationary case, optimal reward changes each step
        # We approximate by looking at reward trends
        cumulative_rewards = np.cumsum(user_rewards)
        base_metrics["final_cumulative_reward"] = float(cumulative_rewards[-1])

        return base_metrics


def get_protocol_class(protocol: Protocol) -> Type[BaseProtocol]:
    """Get the protocol class for a given protocol enum."""
    mapping = {
        Protocol.A_STATIONARY: ProtocolA_Stationary,
        Protocol.B_MODULATED: ProtocolB_Modulated,
        Protocol.C_NON_STATIONARY: ProtocolC_NonStationary,
    }
    return mapping[protocol]


def create_protocol(config: ExperimentConfig) -> BaseProtocol:
    """Create a protocol instance from configuration."""
    protocol_class = get_protocol_class(config.protocol)
    return protocol_class(config)
