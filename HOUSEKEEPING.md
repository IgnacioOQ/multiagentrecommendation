# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md

# Current Project Housekeeping

## Dependency Network
- `src/imports.py`: Base imports.
- `src/agents.py`: Depends on `src.imports`.
- `src/environment.py`: Depends on `src.imports`.
- `src/reward_modulators.py`: Depends on `torch`, `src.imports`, `src.agents`, `tqdm`.
- `src/simulations.py`: Depends on `src.imports`, `src.reward_modulators`, `src.agents`, `src.environment`.
- `src/utils.py`: Depends on `src.imports`.
- `src/stationarity_analysis.py`: Depends on `src.imports`.

## Latest Report

**Execution Date:** 2026-06-25

**Test Results:**

*   **Unit Tests (`tests/`):** PASSED
    - Encountered initial error `ModuleNotFoundError: No module named 'numpy'` across all test files.
    - Fixed by running `pip install -r requirements.txt`.
    - Afterwards, all 13 tests passed successfully.

*   **Notebook Verification (`notebooks/`):**
    - `testing_homeostasis.ipynb`: Executed successfully.
    - `testing_peaks.ipynb`: Executed successfully.
    - `testing_rows.ipynb`: Executed successfully.
    - Converted notebooks using `nbconvert`, replaced `tqdm.notebook` with `tqdm`, commented out `plt.show()`, and reduced `n_steps` for faster verification.

**Summary:**
Housekeeping tasks complete. The dependency graph was mapped. Missing dependencies were installed, clearing initial errors, and all tests and notebook simulations successfully verified without regressions.
**Author:** AI Assistant