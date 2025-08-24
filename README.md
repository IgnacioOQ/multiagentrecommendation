# Human-in-the-Loop Reinforcement Learning Simulation

## Description

This project is a simulation environment designed to study the interactions between a recommender agent and a user agent in a reinforcement learning setting. The simulation allows for the exploration of how different reward modulation strategies can affect the learning process of the agents.

The core of the project is a `run_recommender_simulation` function that simulates the interaction between the two agents in a customizable environment. The simulation can be configured with different parameters, such as the number of recommendations, the number of contexts, the stationarity of the environment, and the type of reward modulation to be used.

## Project Structure

The project is organized into the following main files:

-   `agents.py`: Contains the implementation of the `BaseQLearningAgent`, `RecommenderAgent`, and `RecommendedAgent` classes. These agents use Q-learning to make decisions.
-   `environment.py`: Defines the `ExogenousRewardEnvironment` class, which represents the environment in which the agents interact.
-   `simulations.py`: Contains the main simulation logic, including the `run_recommender_simulation` function.
-   `utils.py`: Provides utility functions for plotting the results of the simulations.
-   `reward_modulators.py`: Contains different classes for modulating the rewards given to the agents, such as `MoodSwings`, `ReceptorModulator`, and `NoveltyModulator`.
-   `stationarity_analysis.py`: Includes functions for analyzing the stationarity of the reward signals.
-   `testing_*.ipynb`: A set of Jupyter notebooks for testing different aspects of the simulation environment.

## How to Run

To run the simulations and tests, you can execute the Python scripts converted from the Jupyter notebooks. For example, to run the tests in `testing_peaks.ipynb`, you can run the following commands:

```bash
jupyter nbconvert --to script testing_peaks.ipynb
python testing_peaks.py
```

This will run the simulations defined in the notebook and generate plots to visualize the results.

## Dependencies

The main dependencies for this project are listed in the `requirements.txt` file.

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```
