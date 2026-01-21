"""
Agents module for the Multi-Agent Recommendation System.

This module exposes all agent implementations for easy import throughout the project.

Usage:
    from src.agents import BaseAgent, DQNAgent, PPOAgent, LinUCBAgent, MCTSAgent
    from src.agents import BaseQLearningAgent, RecommenderAgent, RecommendedAgent
"""

# Base agent interface
from .base import BaseAgent

# Q-Learning agents (tabular)
from .q_learning import BaseQLearningAgent, RecommenderAgent, RecommendedAgent

# Bandit agents
from .bandit import RandomAgent, LinUCBAgent

# Deep RL agents
from .dqn import DQNAgent, ReplayBuffer
from .ppo import PPOAgent

# Planning agents
from .mcts import MCTSAgent, MCTSNode

__all__ = [
    # Base
    "BaseAgent",
    # Q-Learning (tabular)
    "BaseQLearningAgent",
    "RecommenderAgent",
    "RecommendedAgent",
    # Bandits
    "RandomAgent",
    "LinUCBAgent",
    # Deep RL
    "DQNAgent",
    "ReplayBuffer",
    "PPOAgent",
    # Planning
    "MCTSAgent",
    "MCTSNode",
]
