"""
Experiments module for the Multi-Agent Recommendation System.

This module provides a unified interface for running reproducible experiments
to study recommender system dynamics, preference formation, and reward modulation.

Usage:
    from src.experiments import ExperimentConfig, ExperimentRunner
    from src.experiments.config import Protocol, DataSource, ModulatorType

    config = ExperimentConfig(
        name="my_experiment",
        protocol=Protocol.A_STATIONARY,
        n_steps=100000
    )
    runner = ExperimentRunner(config)
    result = runner.run()

For bandit experiments:
    from src.experiments import BanditExperimentRunner

    config = ExperimentConfig(
        name="bandit_exp",
        data_adapter=DataAdapterConfig(data_source=DataSource.AMAZON_BEAUTY)
    )
    runner = BanditExperimentRunner(config)
    result = runner.run_replay_evaluation()
"""

# Configuration
from .config import (
    ExperimentConfig,
    AgentConfig,
    ModulatorConfig,
    EnvironmentConfig,
    MCAnalysisConfig,
    DataAdapterConfig,
    OutputConfig,
    DataSource,
    Protocol,
    AgentType,
    ModulatorType,
)

# Runners
from .runner import ExperimentRunner, BanditExperimentRunner

# Protocols
from .protocols import (
    BaseProtocol,
    ProtocolA_Stationary,
    ProtocolB_Modulated,
    ProtocolC_NonStationary,
    ProtocolResult,
    get_protocol_class,
    create_protocol,
)

# Adapters
from .adapters import (
    BaseDataAdapter,
    MovieLensEnvironmentAdapter,
    AmazonBeautyBanditAdapter,
)

# Utilities
from .utils import (
    compute_config_hash,
    load_experiment_config,
    load_experiment_results,
    load_experiment_metrics,
    compare_experiments,
    find_experiment_dirs,
    create_experiment_batch,
    generate_experiment_name,
)

__all__ = [
    # Configuration
    "ExperimentConfig",
    "AgentConfig",
    "ModulatorConfig",
    "EnvironmentConfig",
    "MCAnalysisConfig",
    "DataAdapterConfig",
    "OutputConfig",
    "DataSource",
    "Protocol",
    "AgentType",
    "ModulatorType",
    # Runners
    "ExperimentRunner",
    "BanditExperimentRunner",
    # Protocols
    "BaseProtocol",
    "ProtocolA_Stationary",
    "ProtocolB_Modulated",
    "ProtocolC_NonStationary",
    "ProtocolResult",
    "get_protocol_class",
    "create_protocol",
    # Adapters
    "BaseDataAdapter",
    "MovieLensEnvironmentAdapter",
    "AmazonBeautyBanditAdapter",
    # Utilities
    "compute_config_hash",
    "load_experiment_config",
    "load_experiment_results",
    "load_experiment_metrics",
    "compare_experiments",
    "find_experiment_dirs",
    "create_experiment_batch",
    "generate_experiment_name",
]
