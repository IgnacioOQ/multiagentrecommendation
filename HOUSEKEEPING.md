# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md.

# Current Project Housekeeping

## Dependency Network

Based on post-React integration analysis:
- **Core Modules:**
- **Advanced Modules:**
- **Engine Module:**
- **API Module:**
- **Tests:**
- **Notebooks:**

## Latest Report

**Execution Date:** 2026-01-19

**Test Results:**
1. `tests/test_api.py`: **Passed** (17 tests).
2. `tests/test_engine.py`: **Passed** (16 tests).
3. `tests/test_mechanics.py`: **Passed** (4 tests).
4. `tests/test_proxy_simulation.py`: **Passed** (1 test).

**Total: 38 tests passed.**

**Summary:**
All unit tests passed. `verify_logging.py` confirmed correct simulation flow and logging. Data persistence features have been integrated and verified locally. Project is stable.
