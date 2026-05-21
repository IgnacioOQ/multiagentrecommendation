---
status: active
type: explanation
description: Overview of the recommender-recommended RL simulation — the two Q-learning agents, reward-modulation components, environment, setup, running simulations, and repository structure.
label: [human]
injection: background
volatility: evolving
scope: project-specific
last_checked: '2026-05-21'
---
# Recommender-Recommended RL Simulation

This project simulates the dynamic interplay between a recommender system and a user, both modeled as reinforcement learning agents. It also uses real world data to inform the simulations. The ultimate goal of the project is to develop an Offline Evaluation (or Offline Experimentation) platform for recommender systems. In particular, it should help identify and explore how internal states and reward modulation can shape the user's learning and interaction over time.

## Core Concepts

The simulation is built around two key agents:

- **Recommender Agent**: This agent learns to suggest items to the user. It receives a positive reward for an accepted recommendation and a negative one for a rejection. Its goal is to maximize accepted recommendations.
- **User Agent (RecommendedAgent)**: This agent learns to accept or reject the recommender's suggestions. It receives a reward from the environment based on the item it's offered. Its goal is to maximize its own reward.

This setup creates a human-in-the-loop system where the agents' decisions mutually influence each other's learning and behavior.

## Key Components

The project is organized into several key modules:

- **`agents.py`**: Defines the `RecommenderAgent` and `RecommendedAgent`. Both are built on a `BaseQLearningAgent` and use Q-learning to adapt their strategies.
- **`environment.py`**: Creates a 2D reward landscape where the x-axis represents different contexts and the y-axis represents different recommendations. The value at each point in the landscape is the reward the user receives for accepting a recommendation in a given context.
- **`simulations.py`**: The core of the project, this module runs the simulation loop, manages the interaction between the agents, and collects data for analysis.
- **`reward_modulators.py`**: This is where the project's most unique features are implemented. These modulators can alter the user's perception of rewards based on different psychological and biological models:
    - **`MoodSwings`**: Simulates fluctuating moods that can unpredictably alter the perceived reward.
    - **`ReceptorModulator`**: Models receptor downregulation, where sensitivity to rewards decreases after repeated exposure.
    - **`NoveltyModulator`**: Adds a bonus for new or infrequent recommendations, encouraging exploration.
    - **`HomeostaticModulator`**: Aims to maintain an internal equilibrium by adjusting rewards to counteract large swings.

## Getting Started

To get started with the simulation, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Simulations

The project includes several Jupyter notebooks and Python scripts for running simulations and tests.

### Testing the `ReceptorModulator`

To test the behavior of the `ReceptorModulator`, you can run the `test_receptor_modulator.py` script:

```bash
python test_receptor_modulator.py
```

This will run a simulation and generate a plot (`receptor_modulator_test.png`) that visualizes how the modulator's sensitivity changes in response to different reward levels.

### Using the Jupyter Notebooks

The project also includes several Jupyter notebooks for more in-depth analysis:

- `testing_homeostasis.ipynb`
- `testing_peaks.ipynb`
- `testing_rows.ipynb`

To run these, you'll need to have Jupyter Notebook installed (`pip install notebook`). Then, you can launch a notebook server from the project's root directory:

```bash
jupyter notebook
```

This will open a new tab in your browser where you can navigate to and run the notebooks.

## Dependencies

All the necessary Python packages are listed in the `requirements.txt` file.

## Project Structure

```text
├── src/                    # Core Python modules
│   ├── agents/             # Q-Learning agent classes
│   ├── utils/              # Utility functions
│   ├── environment.py      # Reward environment
│   ├── simulations.py      # Main simulation runner
│   └── reward_modulators.py # Reward modulation classes
├── tests/                  # Unit and integration tests
├── notebooks/              # Jupyter notebooks for analysis
│   ├── explanation.ipynb
│   ├── testing_homeostasis.ipynb
│   ├── testing_peaks.ipynb
│   └── testing_rows.ipynb
├── data/                   # Data files
├── models/                 # Saved models
├── AI_AGENTS/              # Agent instruction files
│   ├── LINEARIZE_AGENT.md
│   ├── MC_AGENT.md
│   └── RECSYS_AGENT.md
├── HOUSEKEEPING.md         # Recurring sanity-check workflow
├── housekeeping_log.jsonl  # Archived housekeeping reports
├── TODO_WORKFLOW.md        # Cross-session task backlog
├── worklog.jsonl           # Append-only session worklog
├── LICENSE                 # License notice
└── requirements.txt        # Python dependencies
```

## Markdown Conventions

All Markdown files in this repository follow the canonical MDDIA schema (Markdown-JSON Hybrid Schema, Diátaxis Edition): YAML frontmatter at the top of every file, a single `#` title, and the document-type / injection / volatility metadata fields. The specification is maintained centrally in the knowledge base and read via the `kb_mcp` server — this repository does not keep its own `MD_CONVENTIONS.md` copy.
