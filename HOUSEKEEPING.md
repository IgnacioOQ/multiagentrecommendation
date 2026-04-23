# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md

# Current Project Housekeeping

## Dependency Network
- **Core Modules (in `src/`):**
    - `src/imports.py`: Base dependencies (numpy, torch, etc.).
    - `src/utils.py`: Depends on `src.imports`.
    - `src/agents.py`: Depends on `src.imports`.
    - `src/environment.py`: Depends on `src.imports`.
    - `src/reward_modulators.py`: Depends on `src.imports`, `src.agents`, `torch`, `tqdm`.
    - `src/reward_modulators copy.py`: Same as above but with absolute imports.
    - `src/simulations.py`: Depends on `src.imports`, `src.agents`, `src.environment`, `src.reward_modulators`.
    - `src/stationarity_analysis.py`: Depends on `src.imports`.

- **Tests (in `tests/`):**
    - `tests/test_agents.py`: Depends on `src.agents`, `numpy`, `sys`, `os`, `unittest`.
    - `tests/test_environment.py`: Depends on `src.environment`, `numpy`, `sys`, `os`, `unittest`.
    - `tests/test_modulators.py`: Depends on `src.reward_modulators`, `numpy`, `sys`, `os`, `unittest`.
    - `tests/test_receptor_modulator.py`: Depends on `src.reward_modulators`, `tqdm`, `numpy`, `sys`, `os`, `pandas`, `matplotlib.pyplot`.

## Latest Report

**Execution Date:** 2026-04-23
**Author:** AI Agent (Jules)

**Test Results:**

*   **Initial Test Execution (`tests/`):** FAILED
    - `test_agents.py`: `ModuleNotFoundError: No module named 'numpy'`
    - `test_environment.py`: `ModuleNotFoundError: No module named 'numpy'`
    - `test_modulators.py`: `ModuleNotFoundError: No module named 'numpy'`
    - `test_receptor_modulator.py`: `ModuleNotFoundError: No module named 'numpy'`

*   **Resolution:** Installed missing dependencies via `pip install -r requirements.txt`.

*   **Subsequent Unit Tests (`tests/`):** PASSED
    - `test_agents.py`, `test_environment.py`, `test_modulators.py` ran successfully.
    - `test_receptor_modulator.py`: Executed successfully.

*   **Notebook Verification (`notebooks/`):**
    - Processed by converting to scripts via `jupyter nbconvert`, replacing `tqdm.notebook` with `tqdm`, mocking `plt.show()`, and reducing iteration counts.
    - `testing_homeostasis.ipynb`: Executed successfully.
    - `testing_peaks.ipynb`: Executed successfully.
    - `testing_rows.ipynb`: Executed successfully. Generated visualizations without errors.

**Summary:**
Mapped dependencies across the codebase using an AST-based parser. Encountered `ModuleNotFoundError` during initial test execution due to missing dependencies, which was resolved by installing `requirements.txt`. All standard unit tests and converted notebook verification scripts passed following the setup fix.
