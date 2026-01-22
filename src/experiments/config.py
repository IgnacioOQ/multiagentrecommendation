"""
Experiment configuration dataclasses for the Multi-Agent Recommendation System.

This module defines all configuration structures needed to fully specify
and reproduce experiments for studying recommender system dynamics.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List, Tuple
from enum import Enum
import dataclasses
import json


class DataSource(Enum):
    """Available data sources for experiments."""
    SYNTHETIC = "synthetic"
    MOVIELENS = "movielens"
    AMAZON_BEAUTY = "amazon_beauty"


class Protocol(Enum):
    """Experiment protocols as defined in TODOS.md."""
    A_STATIONARY = "stationary_baseline"
    B_MODULATED = "modulated_learning"
    C_NON_STATIONARY = "non_stationary"


class AgentType(Enum):
    """Available agent types."""
    Q_LEARNING = "q_learning"
    LINUCB = "linucb"
    DQN = "dqn"
    RANDOM = "random"


class ModulatorType(Enum):
    """Available reward modulator types."""
    NONE = "none"
    RECEPTOR = "receptor"
    NOVELTY = "novelty"
    HOMEOSTATIC = "homeostatic"
    TD_HOMEOSTATIC = "td_homeostatic"
    MOOD_SWINGS = "mood_swings"


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    agent_type: AgentType = AgentType.Q_LEARNING
    learning_rate: float = 0.1
    gamma: float = 0.9
    exploration_rate: float = 0.1
    exploration_decay: float = 0.999
    min_exploration_rate: float = 0.001
    strategy: Literal["egreedy", "ucb", "softmax"] = "egreedy"
    # LinUCB-specific
    alpha: float = 1.0  # UCB exploration parameter
    regularization: float = 1.0


@dataclass
class ModulatorConfig:
    """Configuration for reward modulation."""
    modulator_type: ModulatorType = ModulatorType.NONE
    # ReceptorModulator parameters
    alpha: float = 0.0001  # desensitization rate
    beta: float = 0.001    # recovery rate
    min_sensitivity: float = 0.1
    max_sensitivity: float = 1.0
    desensitization_threshold: float = 10.0
    # NoveltyModulator parameters
    eta: float = 1.0  # novelty bonus weight
    # HomeostaticModulator parameters
    setpoint: float = 0.0
    lag: int = 0
    n_bins: int = 20
    reward_range: Tuple[float, float] = (-100.0, 100.0)


@dataclass
class EnvironmentConfig:
    """Configuration for the simulation environment."""
    n_recommendations: int = 20  # Y-axis of reward landscape
    n_contexts: int = 50         # X-axis of reward landscape
    landscape_type: Literal["default", "rows"] = "default"
    stationarity: bool = True
    # For non-stationary experiments
    shift_interval: int = 100    # Steps between environment shifts


@dataclass
class MCAnalysisConfig:
    """Configuration for Markov Chain analysis."""
    enabled: bool = False
    snapshot_interval: int = 1000  # Steps between state snapshots
    track_transitions: bool = False
    compute_absorption: bool = False
    n_absorption_simulations: int = 100


@dataclass
class DataAdapterConfig:
    """Configuration for data adapters."""
    data_source: DataSource = DataSource.SYNTHETIC
    data_dir: str = "data"
    # MovieLens adapter
    top_n_movies: int = 20           # Number of movies to use as recommendations
    user_clustering_method: Literal["genre", "temporal", "kmeans"] = "genre"
    n_user_clusters: int = 50        # Map to n_contexts
    # Amazon adapter
    top_n_products: int = 50         # Number of products as arms
    context_dim: int = 100           # TF-IDF feature dimension
    reward_threshold: float = 4.0    # Rating >= threshold is positive


@dataclass
class OutputConfig:
    """Configuration for output and logging."""
    output_dir: str = "outputs"
    save_results: bool = True
    save_figures: bool = True
    save_checkpoints: bool = False
    checkpoint_interval: int = 100000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    This is the main configuration object passed to ExperimentRunner.
    It contains all parameters needed to fully specify and reproduce an experiment.
    """
    # Experiment metadata
    name: str = "experiment"
    description: str = ""
    protocol: Protocol = Protocol.A_STATIONARY

    # Reproducibility
    random_seed: int = 42

    # Simulation parameters
    n_steps: int = 1000000
    rolling_window: int = 1000

    # Initialize agents with prior knowledge
    initialize_recommender: bool = True
    initialize_recommended: bool = True

    # Component configurations
    recommender_agent: AgentConfig = field(default_factory=AgentConfig)
    recommended_agent: AgentConfig = field(default_factory=AgentConfig)
    modulator: ModulatorConfig = field(default_factory=ModulatorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data_adapter: DataAdapterConfig = field(default_factory=DataAdapterConfig)
    mc_analysis: MCAnalysisConfig = field(default_factory=MCAnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        return convert(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        def reconstruct(data, target_cls):
            if not dataclasses.is_dataclass(target_cls):
                return data

            field_types = {f.name: f.type for f in dataclasses.fields(target_cls)}
            kwargs = {}

            for key, value in data.items():
                if key not in field_types:
                    continue

                field_type = field_types[key]

                # Handle Enum types
                if isinstance(field_type, type) and issubclass(field_type, Enum):
                    kwargs[key] = field_type(value)
                # Handle nested dataclasses
                elif dataclasses.is_dataclass(field_type):
                    kwargs[key] = reconstruct(value, field_type)
                # Handle tuple (stored as list)
                elif hasattr(field_type, '__origin__') and field_type.__origin__ is tuple:
                    kwargs[key] = tuple(value) if isinstance(value, list) else value
                else:
                    kwargs[key] = value

            return target_cls(**kwargs)

        return reconstruct(d, cls)

    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []

        if self.n_steps < 1000:
            warnings.append("n_steps < 1000: May be insufficient for learning")

        if self.protocol == Protocol.B_MODULATED and self.modulator.modulator_type == ModulatorType.NONE:
            warnings.append("Protocol B selected but no modulator configured")

        if self.protocol == Protocol.C_NON_STATIONARY and self.environment.stationarity:
            warnings.append("Protocol C selected but stationarity=True")

        if self.protocol == Protocol.A_STATIONARY and self.modulator.modulator_type != ModulatorType.NONE:
            warnings.append("Protocol A selected but modulator is configured (will be ignored)")

        if self.data_adapter.data_source == DataSource.MOVIELENS:
            if self.data_adapter.top_n_movies != self.environment.n_recommendations:
                warnings.append(
                    f"MovieLens top_n_movies ({self.data_adapter.top_n_movies}) differs from "
                    f"n_recommendations ({self.environment.n_recommendations})"
                )
            if self.data_adapter.n_user_clusters != self.environment.n_contexts:
                warnings.append(
                    f"MovieLens n_user_clusters ({self.data_adapter.n_user_clusters}) differs from "
                    f"n_contexts ({self.environment.n_contexts})"
                )

        return warnings

    def copy(self, **overrides) -> "ExperimentConfig":
        """Create a copy of this config with optional overrides."""
        d = self.to_dict()

        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like "modulator.alpha"
                parts = key.split('.')
                target = d
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
            else:
                d[key] = value

        return self.from_dict(d)
