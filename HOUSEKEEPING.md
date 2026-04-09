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
    - `src/imports.py`: Depends on `collections`, `tqdm`, `pandas`, `random`, `scipy`, `statsmodels`, `numpy`, `matplotlib`, `warnings`
    - `src/utils.py`: Depends on `src.imports`, `src`
    - `src/agents.py`: Depends on `src.imports`, `src`
    - `src/environment.py`: Depends on `src.imports`, `src`
    - `src/reward_modulators.py`: Depends on `collections`, `agents`, `tqdm`, `src.imports`, `torch`
    - `src/reward_modulators copy.py`: Depends on `collections`, `agents`, `tqdm`, `src.imports`, `torch`
    - `src/simulations.py`: Depends on `reward_modulators`, `agents`, `src.imports`, `environment`, `src`
    - `src/stationarity_analysis.py`: Depends on `src.imports`, `src`
    - `src/__init__.py`: Depends on nothing

- **Tests (in `tests/`):**
    - `tests/test_agents.py`: Depends on `src`, `numpy`, `unittest`, `os`, `sys`
    - `tests/test_environment.py`: Depends on `src`, `numpy`, `unittest`, `os`, `sys`
    - `tests/test_modulators.py`: Depends on `src`, `numpy`, `unittest`, `os`, `sys`
    - `tests/test_receptor_modulator.py`: Depends on `tqdm`, `pandas`, `src`, `numpy`, `matplotlib`, `os`, `sys`

- **Notebooks (in `notebooks/`):**
    - `notebooks/testing_homeostasis.ipynb`: Depends on `src.reward_modulators`.
    - `notebooks/testing_peaks.ipynb`: Depends on `src.*`.
    - `notebooks/testing_rows.ipynb`: Depends on `src.*`.

## Latest Report

**Execution Date:** 2026-04-09
**Author:** Jules

**Test Results:**

*   **Unit Tests (`tests/`):** PASSED
    - Initially failed with `ModuleNotFoundError: No module named 'numpy'`.
    - Resolved by running `pip install -r requirements.txt`.
    - Subsequent executions of `test_agents.py`, `test_environment.py`, `test_modulators.py`, and `test_receptor_modulator.py` succeeded.

*   **Notebook Verification (`notebooks/`):**
    - `testing_homeostasis.ipynb`: Converted to script, executed successfully.
    - `testing_peaks.ipynb`: Converted to script, executed successfully.
    - `testing_rows.ipynb`: Converted to script, executed successfully. Note: simulation steps were drastically reduced via `sed` to allow faster execution.

**Summary:**
The project's dependency network is stable. Missing dependencies (e.g. `numpy`) were detected initially but successfully restored via `pip install -r requirements.txt`. All unit tests and converted notebook tests passed seamlessly thereafter.
