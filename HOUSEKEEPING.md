# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Add that report to the AGENTS_LOG.md
5. Compile all errors and tests results into a report. And print that report in the Latest Report subsection below, overwriting previous reports.

# Current Project Housekeeping

## Dependency Network

Based on updated import analysis (including advanced modules):
- **Core Modules:**
  - `src/logging.py`: Standard libs, numpy.
  - `src/analysis.py`: numpy.
  - `src/environment.py`: numpy.
  - `src/agents.py`: numpy, torch, random, collections.
  - `src/simulation.py`: Depends on `src.agents`, `src.environment`, `src.logging`, `src.analysis`.
- **Advanced Modules:**
  - `src/advanced_environment.py`: Inherits `src.environment`.
  - `src/advanced_agents.py`: Inherits `src.agents`.
  - `src/advanced_analysis.py`: Depends on numpy, torch.
  - `src/advanced_simulation.py`: Depends on advanced modules + `src.simulation` + `src.logging`.
- **Tests:**
  - `tests/test_mechanics.py`: Covers core modules.
  - `tests/advanced_test_mechanics.py`: Covers advanced modules.
- **Notebooks:**
  - `notebooks/experiment_interface.ipynb`: Uses `src.simulation`.
  - `notebooks/advanced_experiment_interface.ipynb`: Uses `src.advanced_simulation`.

## Latest Report

**Execution Date:** 2024-05-22

**Test Results:**
1. `tests/test_mechanics.py`: **Passed** (4 tests).
2. `tests/advanced_test_mechanics.py`: **Passed** (5 tests).

**Summary:**
All core and advanced component tests passed. The dependency graph shows clear inheritance and modular extension for the advanced features.
