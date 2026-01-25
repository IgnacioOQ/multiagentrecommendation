import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Multi-Agent Recommender System: Concepts & Architecture\n",
    "\n",
    "This notebook provides a pedagogical explanation of the core components of the project:\n",
    "1. **Data & Simulations**: How real-world data feeds into the simulation environment.\n",
    "2. **Experiment Protocols**: The standardized ways we test our hypotheses.\n",
    "3. **Markov Chain Analysis**: How we analyze the long-term behavior of the system.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Connecting Simulations and Data\n",
    "\n",
    "In this project, we bridge the gap between static datasets (like MovieLens) and dynamic Reinforcement Learning (RL) simulations using **Adapters**.\n",
    "\n",
    "### The Concept\n",
    "A typical RL simulation needs an **Environment** that provides rewards. Instead of using a purely synthetic environment, we create a **Reward Landscape** derived from real data.\n",
    "\n",
    "*   **X-Axis (Contexts)**: We cluster users into groups based on their behavior (e.g., genre preferences). Each cluster becomes a \"Context\".\n",
    "*   **Y-Axis (Recommendations)**: We select the top-N most popular items (e.g., movies). These are the actions available to the Recommender.\n",
    "*   **Z-Axis (Reward)**: The average rating given by users in a specific cluster to a specific item becomes the expected reward.\n",
    "\n",
    "### Visualization\n",
    "The code below generates a synthetic \"Reward Landscape\" to illustrate what the `MovieLensEnvironmentAdapter` constructs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Simulate a Reward Landscape associated with MovieLensAdapter\n",
    "n_contexts = 20  # User Clusters (e.g., \"Action Lovers\", \"Romance Fans\")\n",
    "n_items = 10     # Top Movies\n",
    "\n",
    "# Create a synthetic landscape\n",
    "# Represents average ratings (normalized to -1 to 1 for the simulation)\n",
    "landscape = np.random.uniform(-0.5, 0.5, (n_items, n_contexts))\n",
    "\n",
    "# Add some structure (Action lovers like Action movies)\n",
    "# Context 0-5 (Action Fans) like Items 0-3 (Action Movies)\n",
    "landscape[0:3, 0:5] += 0.8\n",
    "\n",
    "# Context 15-20 (Romance Fans) like Items 7-10 (Romance Movies)\n",
    "landscape[7:10, 15:20] += 0.8\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.heatmap(landscape, cmap=\"coolwarm\", center=0, \n",
    "            xticklabels=[f\"Cluster {i}\" for i in range(n_contexts)],\n",
    "            yticklabels=[f\"Movie {i}\" for i in range(n_items)])\n",
    "plt.title(\"Conceptual Reward Landscape (Derived from Data)\")\n",
    "plt.xlabel(\"User Contexts (Clusters)\")\n",
    "plt.ylabel(\"Recommendations (Items)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Experiment Protocols\n",
    "\n",
    "To ensure scientific rigor, we defined three standard protocols in `src/experiments/protocols.py`. Each protocol tests a specific aspect of the User-Recommender interaction.\n",
    "\n",
    "| Protocol | Name | Description | Key Metric |\n",
    "| :--- | :--- | :--- | :--- |\n",
    "| **A** | **Stationary Baseline** | Standard RL setup. No psychological modulation. Fixed environment. | **Convergence Time**: How fast does the agent find the optimal items? |\n",
    "| **B** | **Modulated Learning** | Initializes psychological modulators (e.g., Boredom). The user's reward perception changes based heavily on history. | **Suboptimality Gap**: Does the user get \"stuck\" accepting worse items because of their internal state? |\n",
    "| **C** | **Non-Stationary** | The environment itself changes over time (e.g., user preferences drift naturally). | **Tracking Error**: Can the recommender keep up with the changing user? |\n",
    "\n",
    "### Example Configuration\n",
    "Here is how these protocols are configured in code:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from dataclasses import dataclass\n",
    "\n",
    "# Simplified representation of the Protocol Config structure\n",
    "@dataclass\n",
    "class ProtocolConfig:\n",
    "    name: str\n",
    "    modulator_active: bool\n",
    "    environment_static: bool\n",
    "\n",
    "protocols = {\n",
    "    \"A\": ProtocolConfig(\"Stationary\", modulator_active=False, environment_static=True),\n",
    "    \"B\": ProtocolConfig(\"Modulated\", modulator_active=True, environment_static=True),\n",
    "    \"C\": ProtocolConfig(\"Non-Stationary\", modulator_active=False, environment_static=False)\n",
    "}\n",
    "\n",
    "for key, p in protocols.items():\n",
    "    print(f\"Protocol {key}: {p.name} -> Modulator: {p.modulator_active}, Static Env: {p.environment_static}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Markov Chain Analysis\n",
    "\n",
    "We model the entire interaction as a **Markov Chain**. \n",
    "\n",
    "### What is the \"State\"?\n",
    "The state of the system is complex. It represents the combination of:\n",
    "1.  **User's Beliefs**:Represented by their Q-values (what they think is good).\n",
    "2.  **Recommender's Beliefs**: Represented by their Q-values (what they think the user likes).\n",
    "3.  **Modulator State**: E.g., current \"boredom\" level.\n",
    "4.  **Context**: The current user cluster active in the environment.\n",
    "\n",
    "### Why Markov Chains?\n",
    "By treating simulations as trajectories through this state space, we can mathematically analyze:\n",
    "*   **Absorption**: Does the system settle into a permanent loop (e.g., Filter Bubble)?\n",
    "*   **Mixing Time**: How long generally does it take to stabilize?\n",
    "*   **Ergodicity**: Can the system reach any state from any other state, or are some regions \"walled off\"?\n",
    "\n",
    "### Visualization: Convergence Trajectory\n",
    "We track the \"distance\" between states over time. As the agents learn, the system state should change less and less, eventually approaching zero (Absorption)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Simulated convergence data\n",
    "steps = np.arange(0, 1000, 10)\n",
    "# Distance between consecutive states decreases as agents learn\n",
    "state_deltas = np.exp(-steps / 200) + np.random.normal(0, 0.05, len(steps))\n",
    "state_deltas = np.clip(state_deltas, 0, None)\n",
    "\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(steps, state_deltas, label=\"State Change Magnitude\")\n",
    "plt.axhline(0, color='black', linewidth=0.5)\n",
    "plt.axhline(0.1, color='red', linestyle='--', label=\"Absorption Threshold\")\n",
    "plt.title(\"Markov Chain Convergence: State Stability over Time\")\n",
    "plt.xlabel(\"Simulation Step\")\n",
    "plt.ylabel(\"Distance between State(t) and State(t-1)\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("notebooks/explanation.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=2)

print("Notebook created successfully.")
