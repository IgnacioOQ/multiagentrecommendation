# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md

# Current Project Housekeeping

## Dependency Network

Based on the updated `src/` package structure:

- **Core Modules (in `src/`):**
    - `src/imports.py`: Base dependencies (numpy, torch, etc.).
    - `src/utils.py`: Depends on `src.imports`.
    - `src/agents.py`: Depends on `src.imports`.
    - `src/environment.py`: Depends on `src.imports`.
    - `src/reward_modulators.py`: Depends on `src.imports`, `src.agents`.
    - `src/simulations.py`: Depends on `src.imports`, `src.agents`, `src.environment`, `src.reward_modulators`.
    - `src/stationarity_analysis.py`: Depends on `src.imports`.

- **Tests (in `tests/`):**
    - `tests/test_agents.py`: Depends on `src.agents`.
    - `tests/test_environment.py`: Depends on `src.environment`.
    - `tests/test_modulators.py`: Depends on `src.reward_modulators`.
    - `tests/test_receptor_modulator.py`: Depends on `src.reward_modulators`.

- **Notebooks (in `notebooks/`):**
    - `notebooks/testing_homeostasis.ipynb`: Depends on `src.reward_modulators`.
    - `notebooks/testing_peaks.ipynb`: Depends on `src.*`.
    - `notebooks/testing_rows.ipynb`: Depends on `src.*`.

## Latest Report

**Execution Date:** 2025-02-18

**Author:** AI Assistant

**Test Results:**

*   **Unit Tests (`tests/`):** PASSED
    - 13 total tests ran from `test_agents.py`, `test_environment.py`, `test_modulators.py`.
    - `test_receptor_modulator.py`: Verified visual output execution.

*   **Notebook Verification (`notebooks/`):**
    - `testing_homeostasis.ipynb`: Executed successfully without blocking GUI elements.
    - `testing_peaks.ipynb`: Executed successfully without blocking GUI elements.
    - `testing_rows.ipynb`: Executed successfully without blocking GUI elements. Steps were reduced for verification.

**Summary:**
The project's dependency network is stable and follows relative imports inside the `src` module. All unit tests passed successfully. Jupyter notebooks were successfully converted to headless scripts and executed, verifying the integration of agents, environment, and reward modulators.
