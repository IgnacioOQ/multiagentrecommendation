"""
Experiment runners for the Multi-Agent Recommendation System.

Provides:
- ExperimentRunner: Main orchestrator for grid-based Q-learning experiments
- BanditExperimentRunner: Specialized runner for LinUCB/bandit experiments
"""

import os
import random
import logging
from typing import Dict, Any, Optional, Type

import numpy as np

from .config import ExperimentConfig, DataSource, ModulatorType
from .adapters import MovieLensEnvironmentAdapter, AmazonBeautyBanditAdapter
from .protocols import get_protocol_class, create_protocol, ProtocolResult
from . import utils


class ExperimentRunner:
    """
    Main orchestrator for running experiments.

    Handles:
    1. Configuration validation
    2. Random seed management
    3. Data loading via adapters
    4. Simulation execution
    5. MC Analysis integration
    6. Result storage

    Usage:
        config = ExperimentConfig(name="exp1", protocol=Protocol.A_STATIONARY)
        runner = ExperimentRunner(config)
        results = runner.run()
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.protocol = create_protocol(config)

        self.environment_class: Optional[Type] = None
        self.mc_analyzer = None
        self.simulation_results: Optional[Dict[str, Any]] = None
        self.output_dir: Optional[str] = None

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the experiment."""
        logger = logging.getLogger(f"experiment.{self.config.name}")
        logger.setLevel(getattr(logging, self.config.output.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _set_random_seeds(self) -> None:
        """Set all random seeds for reproducibility."""
        seed = self.config.random_seed
        np.random.seed(seed)
        random.seed(seed)
        self.logger.info(f"Random seeds set to {seed}")

    def _load_environment_class(self) -> Type:
        """Load or create the environment class based on data source."""
        from src.environment import ExogenousRewardEnvironment

        data_source = self.config.data_adapter.data_source

        if data_source == DataSource.SYNTHETIC:
            self.logger.info("Using synthetic environment")
            return ExogenousRewardEnvironment

        elif data_source == DataSource.MOVIELENS:
            self.logger.info("Loading MovieLens data...")
            adapter = MovieLensEnvironmentAdapter(
                data_dir=self.config.data_adapter.data_dir,
                n_recommendations=self.config.environment.n_recommendations,
                n_contexts=self.config.environment.n_contexts,
                clustering_method=self.config.data_adapter.user_clustering_method,
            )
            env_class = adapter.create_environment_class()
            self.logger.info(f"MovieLens adapter info: {adapter.get_info()}")
            return env_class

        elif data_source == DataSource.AMAZON_BEAUTY:
            # Amazon Beauty is for bandit experiments, not grid simulation
            self.logger.warning(
                "Amazon Beauty data is designed for bandit experiments, "
                "not grid simulation. Using synthetic environment."
            )
            return ExogenousRewardEnvironment

        else:
            raise ValueError(f"Unknown data source: {data_source}")

    def _create_modulator_factory(self) -> Type:
        """
        Create a modulator factory class that captures configuration kwargs.

        Returns a class that when instantiated creates the configured modulator.
        """
        modulator_class = self.protocol.get_modulator_class()

        if modulator_class is None:
            # Return a no-op modulator class
            class NoModulator:
                def modify_reward(self, r, *args, **kwargs):
                    return r

                def step(self, *args, **kwargs):
                    pass

            return NoModulator

        mod_kwargs = self.protocol.create_modulator_kwargs()

        # Create a factory class that captures the kwargs
        class ModulatorFactory:
            _base_class = modulator_class
            _kwargs = mod_kwargs

            def __new__(cls):
                return cls._base_class(**cls._kwargs)

        return ModulatorFactory

    def _run_simulation(self) -> Dict[str, Any]:
        """Execute the main simulation loop."""
        from src.simulations import run_recommender_simulation
        from src.agents import RecommenderAgent, RecommendedAgent

        self.logger.info("Starting simulation...")

        modulated = self.config.modulator.modulator_type != ModulatorType.NONE
        modulator_factory = self._create_modulator_factory()

        results = run_recommender_simulation(
            recommender_agent_class=RecommenderAgent,
            recommended_agent_class=RecommendedAgent,
            environment_class=self.environment_class,
            modulator_class=modulator_factory,
            n_recommendations=self.config.environment.n_recommendations,
            n_contexts=self.config.environment.n_contexts,
            n_steps=self.config.n_steps,
            exploration_rate=self.config.recommender_agent.exploration_rate,
            exploration_decay=self.config.recommender_agent.exploration_decay,
            rolling_window=self.config.rolling_window,
            random_seed=self.config.random_seed,
            initialize_recommender=self.config.initialize_recommender,
            initialize_recommended=self.config.initialize_recommended,
            stationarity=self.config.environment.stationarity,
            landscape_type=self.config.environment.landscape_type,
            strategy=self.config.recommended_agent.strategy,
            modulated=modulated,
        )

        self.logger.info(f"Simulation completed: {self.config.n_steps} steps")
        return results

    def _run_mc_analysis(
        self,
        sim_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run Markov Chain analysis if configured."""
        if not self.config.mc_analysis.enabled:
            return None

        self.logger.info("Running Markov Chain analysis...")

        # Note: Full MC analysis requires modifying simulation loop
        # to call snapshot_state() periodically. For now, return
        # basic analysis from the results we have.
        from src.analysis.mc_analysis import MarkovChainAnalyzer

        analyzer = MarkovChainAnalyzer()
        self.mc_analyzer = analyzer

        # Compute what we can from final results
        mc_results = {
            "unique_states": "N/A (requires inline tracking)",
            "absorption_step": None,
            "note": "Full MC analysis requires simulation loop modification",
        }

        return mc_results

    def _save_outputs(
        self,
        results: Dict[str, Any],
        metrics: Dict[str, float],
        mc_analysis: Optional[Dict[str, Any]]
    ) -> None:
        """Save all experiment outputs."""
        if not self.config.output.save_results:
            return

        utils.save_config(self.config, self.output_dir)
        utils.save_results(results, self.output_dir)
        utils.save_metrics(metrics, self.output_dir)

        if mc_analysis:
            import pickle
            mc_path = os.path.join(self.output_dir, "mc_analysis.pkl")
            with open(mc_path, "wb") as f:
                pickle.dump(mc_analysis, f)

        self.logger.info(f"Results saved to {self.output_dir}")

    def run(self) -> ProtocolResult:
        """
        Execute the full experiment pipeline.

        Returns:
            ProtocolResult with metrics, simulation results, and MC analysis
        """
        self.logger.info(f"Starting experiment: {self.config.name}")
        self.logger.info(f"Protocol: {self.config.protocol.value}")

        # Validate configuration
        warnings = self.config.validate()
        for w in warnings:
            self.logger.warning(w)

        self.protocol.validate_config()

        # Set up
        self._set_random_seeds()
        self.output_dir = utils.ensure_output_dir(self.config)

        # Load environment class
        self.environment_class = self._load_environment_class()

        # Run simulation
        self.simulation_results = self._run_simulation()

        # Compute metrics
        metrics = self.protocol.compute_metrics(self.simulation_results)
        self.logger.info(f"Metrics: {metrics}")

        # MC Analysis
        mc_analysis = self._run_mc_analysis(self.simulation_results)

        # Save outputs
        self._save_outputs(self.simulation_results, metrics, mc_analysis)

        # Create result object
        result = ProtocolResult(
            protocol=self.config.protocol,
            metrics=metrics,
            simulation_results=self.simulation_results,
            mc_analysis=mc_analysis,
            output_dir=self.output_dir,
        )

        self.logger.info("Experiment completed successfully")
        return result


class BanditExperimentRunner:
    """
    Specialized runner for LinUCB/bandit experiments with Amazon data.

    Separate from ExperimentRunner because bandit experiments have
    a fundamentally different structure (sequential decisions, not
    grid-based simulation).

    Uses replay (off-policy) evaluation methodology.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.adapter = AmazonBeautyBanditAdapter(
            data_dir=config.data_adapter.data_dir,
            n_arms=config.data_adapter.top_n_products,
            context_dim=config.data_adapter.context_dim,
            reward_threshold=config.data_adapter.reward_threshold,
        )
        self.output_dir: Optional[str] = None

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the experiment."""
        logger = logging.getLogger(f"bandit_experiment.{self.config.name}")
        logger.setLevel(getattr(logging, self.config.output.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run_replay_evaluation(self) -> Dict[str, Any]:
        """
        Run replay (off-policy) evaluation of a bandit policy.

        Uses rejection sampling to evaluate the agent on historical data.
        Only samples where the agent's action matches the logged action
        are used for updating.

        Returns:
            Dictionary with evaluation metrics
        """
        from src.agents import LinUCBAgent

        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)

        self.logger.info("Preparing bandit data...")

        # Prepare data
        contexts, logged_actions, rewards = self.adapter.prepare_bandit_data()
        n_samples = len(contexts)

        self.logger.info(f"Loaded {n_samples} samples for evaluation")

        # Create fresh agent
        agent = LinUCBAgent(
            n_arms=self.config.data_adapter.top_n_products,
            context_dim=self.config.data_adapter.context_dim,
            alpha=self.config.recommender_agent.alpha,
            regularization=self.config.recommender_agent.regularization,
        )

        # Replay evaluation with rejection sampling
        mean_rewards = []
        cumulative_reward = 0.0
        n_matched = 0

        self.logger.info("Running replay evaluation...")

        for i in range(n_samples):
            context = contexts[i]
            logged_action = logged_actions[i]
            reward = rewards[i]

            # Agent selects action
            agent_action = agent.select_action(context, explore=True)

            # Rejection sampling: only update if actions match
            if agent_action == logged_action:
                n_matched += 1
                cumulative_reward += reward

                # Update agent
                agent.store(context, agent_action, reward, context, False)
                agent.update()

                # Track mean reward
                mean_rewards.append(cumulative_reward / n_matched)

        self.logger.info(
            f"Evaluation complete: {n_matched}/{n_samples} matches "
            f"({100*n_matched/n_samples:.1f}%)"
        )

        # Save outputs if configured
        if self.config.output.save_results:
            self.output_dir = utils.ensure_output_dir(self.config)
            utils.save_config(self.config, self.output_dir)

            results = {
                "mean_rewards": mean_rewards,
                "n_matched": n_matched,
                "n_total": n_samples,
                "match_rate": n_matched / n_samples,
                "final_mean_reward": mean_rewards[-1] if mean_rewards else 0.0,
                "adapter_info": self.adapter.get_info(),
            }

            utils.save_results(results, self.output_dir)

            metrics = {
                "n_matched": float(n_matched),
                "n_total": float(n_samples),
                "match_rate": float(n_matched / n_samples),
                "final_mean_reward": float(mean_rewards[-1] if mean_rewards else 0.0),
            }
            utils.save_metrics(metrics, self.output_dir)

            self.logger.info(f"Results saved to {self.output_dir}")

        return {
            "mean_rewards": mean_rewards,
            "n_matched": n_matched,
            "n_total": n_samples,
            "match_rate": n_matched / n_samples,
            "final_mean_reward": mean_rewards[-1] if mean_rewards else 0.0,
            "adapter_info": self.adapter.get_info(),
            "output_dir": self.output_dir,
        }

    def run_online_simulation(
        self,
        n_rounds: int = 10000
    ) -> Dict[str, Any]:
        """
        Run online simulation with the bandit agent.

        Samples contexts from the data and lets the agent choose freely.
        Reward is computed based on the logged data (if available) or
        simulated based on the logged action's reward.

        Args:
            n_rounds: Number of rounds to simulate

        Returns:
            Dictionary with simulation results
        """
        from src.agents import LinUCBAgent

        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)

        # Prepare data
        contexts, logged_actions, rewards = self.adapter.prepare_bandit_data()
        n_samples = len(contexts)

        # Create agent
        agent = LinUCBAgent(
            n_arms=self.config.data_adapter.top_n_products,
            context_dim=self.config.data_adapter.context_dim,
            alpha=self.config.recommender_agent.alpha,
            regularization=self.config.recommender_agent.regularization,
        )

        # Create a reward model from data
        # For simplicity, use the logged reward if action matches,
        # otherwise sample from similar contexts
        reward_history = []
        action_history = []
        cumulative_reward = 0.0

        self.logger.info(f"Running online simulation for {n_rounds} rounds...")

        for round_idx in range(n_rounds):
            # Sample a random context
            idx = np.random.randint(n_samples)
            context = contexts[idx]
            logged_action = logged_actions[idx]
            logged_reward = rewards[idx]

            # Agent selects action
            action = agent.select_action(context, explore=True)

            # Compute reward
            if action == logged_action:
                # We know the true reward
                reward = logged_reward
            else:
                # Simulate: use average reward for this arm
                # This is a simplification
                reward = np.random.binomial(1, 0.5)  # Random baseline

            cumulative_reward += reward
            reward_history.append(reward)
            action_history.append(action)

            # Update agent
            agent.store(context, action, reward, context, False)
            agent.update()

        mean_reward = cumulative_reward / n_rounds

        self.logger.info(f"Simulation complete. Mean reward: {mean_reward:.4f}")

        return {
            "reward_history": reward_history,
            "action_history": action_history,
            "mean_reward": mean_reward,
            "cumulative_reward": cumulative_reward,
            "n_rounds": n_rounds,
        }
