"""
Utility functions for experiment management.

Provides helpers for:
- Configuration hashing and caching
- Loading/saving experiment results
- Comparing multiple experiments
- Creating experiment batches
"""

import os
import json
import pickle
import hashlib
import itertools
from datetime import datetime
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig
    import pandas as pd


def compute_config_hash(config: "ExperimentConfig") -> str:
    """
    Compute a hash of the configuration for caching/deduplication.

    Args:
        config: ExperimentConfig instance

    Returns:
        12-character hex hash string
    """
    config_str = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def load_experiment_config(output_dir: str) -> "ExperimentConfig":
    """
    Load experiment configuration from a previous run.

    Args:
        output_dir: Path to experiment output directory

    Returns:
        ExperimentConfig instance
    """
    from .config import ExperimentConfig

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r") as f:
        return ExperimentConfig.from_dict(json.load(f))


def load_experiment_results(output_dir: str) -> Dict[str, Any]:
    """
    Load results from a previous experiment run.

    Args:
        output_dir: Path to experiment output directory

    Returns:
        Dictionary containing simulation results
    """
    results_path = os.path.join(output_dir, "results.pkl")
    with open(results_path, "rb") as f:
        return pickle.load(f)


def load_experiment_metrics(output_dir: str) -> Dict[str, float]:
    """
    Load metrics from a previous experiment run.

    Args:
        output_dir: Path to experiment output directory

    Returns:
        Dictionary of metric name -> value
    """
    import csv

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return {k: float(v) if v else None for k, v in row.items()}


def compare_experiments(
    dirs: List[str],
    metrics: List[str] = None
) -> "pd.DataFrame":
    """
    Load and compare metrics from multiple experiment runs.

    Args:
        dirs: List of output directories
        metrics: Optional list of specific metrics to compare

    Returns:
        DataFrame with experiments as rows and metrics as columns
    """
    import pandas as pd

    rows = []
    for d in dirs:
        try:
            config_path = os.path.join(d, "config.json")
            metrics_path = os.path.join(d, "metrics.csv")

            with open(config_path) as f:
                config = json.load(f)

            m = pd.read_csv(metrics_path).iloc[0].to_dict()
            m["experiment_name"] = config.get("name", os.path.basename(d))
            m["protocol"] = config.get("protocol", "unknown")
            m["output_dir"] = d
            rows.append(m)
        except Exception as e:
            print(f"Warning: Could not load experiment from {d}: {e}")

    df = pd.DataFrame(rows)

    if metrics and len(df) > 0:
        base_cols = ["experiment_name", "protocol", "output_dir"]
        cols = base_cols + [c for c in metrics if c in df.columns]
        df = df[cols]

    return df


def find_experiment_dirs(
    base_dir: str = "outputs",
    name_pattern: str = None,
    protocol: str = None
) -> List[str]:
    """
    Find experiment output directories matching criteria.

    Args:
        base_dir: Base output directory to search
        name_pattern: Optional substring to match in experiment name
        protocol: Optional protocol to filter by

    Returns:
        List of matching output directory paths
    """
    if not os.path.exists(base_dir):
        return []

    dirs = []
    for entry in os.listdir(base_dir):
        exp_dir = os.path.join(base_dir, entry)
        config_path = os.path.join(exp_dir, "config.json")

        if not os.path.isfile(config_path):
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)

            if name_pattern and name_pattern not in config.get("name", ""):
                continue

            if protocol and config.get("protocol") != protocol:
                continue

            dirs.append(exp_dir)
        except Exception:
            continue

    return sorted(dirs)


def create_experiment_batch(
    base_config: "ExperimentConfig",
    variations: Dict[str, List[Any]]
) -> List["ExperimentConfig"]:
    """
    Create a batch of experiment configurations by varying parameters.

    Args:
        base_config: Base configuration to modify
        variations: Dict mapping parameter paths to lists of values
            e.g., {"modulator.eta": [0.5, 1.0, 2.0]}

    Returns:
        List of ExperimentConfig objects
    """
    configs = []
    keys = list(variations.keys())
    value_lists = [variations[k] for k in keys]

    for values in itertools.product(*value_lists):
        config_dict = base_config.to_dict()

        for key, value in zip(keys, values):
            # Navigate nested dict
            parts = key.split(".")
            d = config_dict
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = value

        # Create name suffix from varied parameters
        suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, values))
        config_dict["name"] = f"{base_config.name}_{suffix}"

        from .config import ExperimentConfig
        new_config = ExperimentConfig.from_dict(config_dict)
        configs.append(new_config)

    return configs


def generate_experiment_name(
    base_name: str,
    include_timestamp: bool = True,
    include_hash: bool = False,
    config: "ExperimentConfig" = None
) -> str:
    """
    Generate a unique experiment name.

    Args:
        base_name: Base name for the experiment
        include_timestamp: Whether to append timestamp
        include_hash: Whether to append config hash
        config: Config to hash (required if include_hash=True)

    Returns:
        Generated experiment name
    """
    parts = [base_name]

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)

    if include_hash and config is not None:
        hash_str = compute_config_hash(config)
        parts.append(hash_str)

    return "_".join(parts)


def ensure_output_dir(config: "ExperimentConfig") -> str:
    """
    Create and return the output directory for an experiment.

    Args:
        config: Experiment configuration

    Returns:
        Path to the output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        config.output.output_dir,
        f"{config.name}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    return output_dir


def save_config(config: "ExperimentConfig", output_dir: str) -> str:
    """
    Save experiment configuration to JSON.

    Args:
        config: Experiment configuration
        output_dir: Output directory path

    Returns:
        Path to saved config file
    """
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    return config_path


def save_results(results: Dict[str, Any], output_dir: str) -> str:
    """
    Save simulation results to pickle.

    Args:
        results: Simulation results dictionary
        output_dir: Output directory path

    Returns:
        Path to saved results file
    """
    results_path = os.path.join(output_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    return results_path


def save_metrics(metrics: Dict[str, float], output_dir: str) -> str:
    """
    Save metrics to CSV.

    Args:
        metrics: Dictionary of metric name -> value
        output_dir: Output directory path

    Returns:
        Path to saved metrics file
    """
    import csv

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics_path
