# Housekeeping Protocol
- status: active

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. The report must include the following fields: **Date**, **Summary**, and **Author** (stating the name of the AI assistant). Print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md.

# Current Project Housekeeping
- status: active

## Dependency Network
- status: active

Based on current codebase analysis:
- **Core Modules:** `src/imports.py`, `src/utils/`
- **Agents:** `src/agents/` (depends on Core)
- **Environment:** `src/environment.py` (depends on Core)
- **Simulation:** `src/simulations.py` (depends on Agents, Environment, Reward Modulators)
- **Data:** `src/data/` (depends on `requests`, `pandas`, `tqdm`)
- **Tests:** `tests/` (depends on all above)

## Latest Report
- status: active

**Date:** 2026-02-18
**Author:** Jules

**Test Results:**
1. `tests/test_download_mock.py`: **Passed** (2 tests). Verified mock downloading pipelines.
2. `tests/test_integration.py`: **Passed** (1 test). Verified data processing integration.
3. `tests/test_receptor_modulator.py`: **Not Run** (Script-based visual test, no assertions).

**Total: 3 tests passed.**

**Summary:**
The directory structure in `AGENTS.md` has been updated to reflect the actual codebase. Unit tests for data downloading and integration passed after installing missing dependencies (`pandas`, `requests`, `torch`, etc.). The project environment is now stable.
