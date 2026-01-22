# Recommender-Recommended RL Simulation
- status: active

This project simulates the dynamic interplay between a recommender system and a user, both modeled as reinforcement learning agents. It provides a flexible framework to explore how internal states and reward modulation can shape their learning and interaction over time.

## Core Concepts
- status: active

The simulation is built around two key agents:

*   **Recommender Agent**: This agent learns to suggest items to the user. It receives a positive reward for an accepted recommendation and a negative one for a rejection. Its goal is to maximize accepted recommendations.
*   **User Agent (RecommendedAgent)**: This agent learns to accept or reject the recommender's suggestions. It receives a reward from the environment based on the item it's offered. Its goal is to maximize its own reward.

This setup creates a human-in-the-loop system where the agents' decisions mutually influence each other's learning and behavior.

## Key Components
- status: active

The project is organized into several key modules:

*   **`agents.py`**: Defines the `RecommenderAgent` and `RecommendedAgent`. Both are built on a `BaseQLearningAgent` and use Q-learning to adapt their strategies.
*   **`environment.py`**: Creates a 2D reward landscape where the x-axis represents different contexts and the y-axis represents different recommendations. The value at each point in the landscape is the reward the user receives for accepting a recommendation in a given context.
*   **`simulations.py`**: The core of the project, this module runs the simulation loop, manages the interaction between the agents, and collects data for analysis.
*   **`reward_modulators.py`**: This is where the project's most unique features are implemented. These modulators can alter the user's perception of rewards based on different psychological and biological models:
    *   **`MoodSwings`**: Simulates fluctuating moods that can unpredictably alter the perceived reward.
    *   **`ReceptorModulator`**: Models receptor downregulation, where sensitivity to rewards decreases after repeated exposure.
    *   **`NoveltyModulator`**: Adds a bonus for new or infrequent recommendations, encouraging exploration.
    *   **`HomeostaticModulator`**: Aims to maintain an internal equilibrium by adjusting rewards to counteract large swings.

## Getting Started
- status: active

To get started with the simulation, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Simulations
- status: active

The project includes several Jupyter notebooks and Python scripts for running simulations and tests.

### Testing the `ReceptorModulator`
- status: active

To test the behavior of the `ReceptorModulator`, you can run the `test_receptor_modulator.py` script:

```bash
python test_receptor_modulator.py
```

This will run a simulation and generate a plot (`receptor_modulator_test.png`) that visualizes how the modulator's sensitivity changes in response to different reward levels.

### Using the Jupyter Notebooks
- status: active

The project also includes several Jupyter notebooks for more in-depth analysis:

*   `testing_homeostasis.ipynb`
*   `testing_peaks.ipynb`
*   `testing_rows.ipynb`

To run these, you'll need to have Jupyter Notebook installed (`pip install notebook`). Then, you can launch a notebook server from the project's root directory:

```bash
jupyter notebook
```

This will open a new tab in your browser where you can navigate to and run the notebooks.

## Dependencies
- status: active

All the necessary Python packages are listed in the `requirements.txt` file.
