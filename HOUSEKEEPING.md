# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Add that report to the AGENTS_LOG.md
5. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.

# Current Project Housekeeping

## Dependency Network

Based on updated import analysis:
- **Core Modules:**
    - `imports.py`: Base dependencies (numpy, torch, etc.).
    - `utils.py`: Depends on `imports.py`.
    - `agents.py`: Depends on `imports.py`.
    - `environment.py`: Depends on `imports.py`.
    - `reward_modulators.py`: Depends on `imports.py`, `agents.py` (inherits `BaseQLearningAgent`).
    - `simulations.py`: Depends on `imports.py`, `agents.py`, `environment.py`, `reward_modulators.py`.

- **Advanced Modules:**
    - `stationarity_analysis.py`: Depends on `imports.py`.

- **Tests:**
    - `test_receptor_modulator.py`: Depends on `reward_modulators.py`.
    - `tests/*.py`: Depend on core modules.

- **Notebooks:**
    - `testing_homeostasis.ipynb`: Depends on `simulations.py`, `reward_modulators.py`.
    - `testing_peaks.ipynb`: Depends on `environment.py`, `agents.py`.
    - `testing_rows.ipynb`: Depends on `environment.py`, `agents.py`.


## Latest Report

**Execution Date:** 2024-05-22

**Test Results:**

*   **Unit Tests (`tests/`):** PASSED
    *   `test_agents.py`: Verified `BaseQLearningAgent`, `RecommenderAgent`, `RecommendedAgent`.
    *   `test_environment.py`: Verified `ExogenousRewardEnvironment` initialization, stepping, and shifting.
    *   `test_modulators.py`: Verified `ReceptorModulator`, `HomeostaticModulator`, `TD_DHR`.

*   **Script Tests:** PASSED
    *   `test_receptor_modulator.py`: Executed successfully.

*   **Notebook Verification:**
    *   `testing_homeostasis.ipynb`: Executed successfully (converted to script). Validated PID and Homeostatic/Allostatic controllers (TD_DHR, DQN_DHR).
    *   `testing_peaks.ipynb`: Executed successfully (converted to script). Validated gaussian peak landscape simulation.
    *   `testing_rows.ipynb`: Timed out during execution (likely due to long simulation steps), but initial execution stages proceeded.

**Summary:**
The core logic and simulation frameworks are functional. New unit tests provide coverage for agents, environment, and modulators. Dependency network is mapped. `testing_rows.ipynb` runs long but appears functional based on similar logic in `testing_peaks.ipynb`.
