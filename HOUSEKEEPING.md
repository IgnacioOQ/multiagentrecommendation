# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md

# Current Project Housekeeping

## Dependency Network

Based on actual parsed imports:

- **Core Modules (in `src/`):**
    - `src/__init__.py`: Depends on None
    - `src/imports.py`: Depends on None
    - `src/utils.py`: Depends on `.imports`
    - `src/agents.py`: Depends on `.imports`
    - `src/environment.py`: Depends on `.imports`
    - `src/reward_modulators copy.py`: Depends on None
    - `src/reward_modulators.py`: Depends on `.agents`, `.imports`
    - `src/simulations.py`: Depends on `.agents`, `.environment`, `.imports`, `.reward_modulators`
    - `src/stationarity_analysis.py`: Depends on `.imports`

- **Tests (in `tests/`):**
    - `tests/test_agents.py`: Depends on `src.agents`
    - `tests/test_environment.py`: Depends on `src.environment`
    - `tests/test_modulators.py`: Depends on `src.reward_modulators`
    - `tests/test_receptor_modulator.py`: Depends on `src.reward_modulators`

## Latest Report

**Execution Date:** 2026-05-14
**Author:** Jules

**Test Results:**

*   **Unit Tests (`tests/`):** PASSED (after initial pip install fix)
    - Encountered initial errors: `ModuleNotFoundError: No module named 'numpy'` in all test files.
    - Fixed by installing dependencies from `requirements.txt`.
    - All 13 tests passed successfully.

*   **Notebook Verification (`notebooks/`):** PASSED
    - Converted notebooks (`testing_homeostasis.ipynb`, `testing_peaks.ipynb`, `testing_rows.ipynb`) to scripts.
    - Modified them to remove `plt.show()` and `tqdm.notebook`.
    - Reduced `num_steps` and `n_steps` to 100-1000 for verification.
    - All 3 notebook scripts ran to completion without errors.
    - Cleaned up generated intermediate scripts.

**Summary:**
Performed housekeeping protocol. Identified and fixed a missing dependency issue during initial test execution (`numpy`). After environment setup, all unit tests and notebook visual verifications passed successfully. The dependency network continues to accurately reflect the correct module hierarchy.
