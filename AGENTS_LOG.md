# Agents Log
- status: active

## Intervention History
- status: active

### Housekeeping Report (Initial)
- status: active
**Date:** 
**Summary:** Executed initial housekeeping protocol.
**AI Assitant:**
- **Dependency Network:** 
- **Tests:** 

### Bug Fix: Advanced Analysis (Shape Mismatch)
- status: active
**Date:** 2024-05-22
**Summary:** Fixed RuntimeError in `advanced_experiment_interface.ipynb`.
- **Issue:** `compute_policy_metrics` in `src/analysis.py` passed 1D inputs `(100, 1)` to agents expecting 2D inputs `(100, 2)`.
- **Fix:** Created `src/advanced_analysis.py` with `compute_advanced_policy_metrics`.
- **Details:** The new function constructs inputs as `[p, t]` with `t` fixed at 0 (default).
- **Files Modified:** `src/advanced_simulation.py` updated to use the new analysis function.

### Bug Fix: Notebook NameError
- status: active
**Date:** 2024-05-22
**Summary:** Fixed NameError in `advanced_experiment_interface.ipynb`.
- **Issue:** The variable `ep_id` was used in a print statement but was undefined in the new JSON saving block.
- **Fix:** Removed the erroneous print statement and cleanup old comments. Validated that the correct logging uses `current_step_info['episode_count']`.

### Project Reorganization
- status: active
**Date:** 2026-01-21
**Summary:** Reorganized project into standard directory structure.
- **Changes:**
    - Created `src/` directory for Python source files: `agents.py`, `environment.py`, `simulations.py`, `reward_modulators.py`, `utils.py`, `stationarity_analysis.py`, `imports.py`
    - Created `tests/` directory for test files: `test_receptor_modulator.py`
    - Created `notebooks/` directory for Jupyter notebooks: `testing_peaks.ipynb`, `testing_homeostasis.ipynb`, `testing_rows.ipynb`
    - Created `AI_AGENTS/` directory for specialized agent instructions: `MC_AGENT.md`
    - Updated all imports to use `src.` prefix (e.g., `from src.imports import *`)
    - Added `src/__init__.py` to make src a proper Python package
    - Removed backup file `reward_modulators copy.py` and root `__pycache__/`
    - Updated AGENTS.md with new directory structure documentation

### Model Training & Bandit Agent Enhancement
- status: active
**Date:** 2026-01-22
**Summary:** Verified model training pipelines and added ContextualBanditAgent.
- **Changes:**
    - **CF Training Verified:** SVD model trained on MovieLens 100K (RMSE: 0.8729, MAE: 0.6712)
    - **Bandit Training Updated:** Replaced `contextualbandits` library dependency with custom `LinUCBAgent` implementation
    - **New Agent:** Added `ContextualBanditAgent` to `src/agents/bandit.py` supporting:
        - Epsilon-greedy exploration (with decay)
        - UCB (Upper Confidence Bound) exploration
        - Softmax (Boltzmann) exploration
    - **Files Modified:**
        - `src/models/train_bandit.py`: Now uses custom `LinUCBAgent` with replay evaluation
        - `src/agents/bandit.py`: Added `ContextualBanditAgent` and `ExplorationStrategy` enum
        - `src/agents/__init__.py`: Exported new classes
        - `requirements.txt`: Added `scikit-surprise` and `scikit-learn`
    - **Visualization Verified:** All 6 plotting functions in `src/plotting_utils.py` tested and working
    - **TODOS.md Updated:** Phase 1 now fully complete with documented baseline metrics

### Documentation Update & Housekeeping
- status: active
**Date:** 2026-02-18
**Summary:** Updated AGENTS.md to match codebase and executed housekeeping protocol.
- **Changes:**
    - Updated `AGENTS.md` to remove incorrect "Monorepo", "Backend", "Frontend" references and correct the directory structure documentation (src/agents/ directory, specific test files).
    - Executed `HOUSEKEEPING.md` protocol:
        - Analyzed dependency network.
        - Installed missing dependencies (`pandas`, `requests`, `torch`, `statsmodels`, etc.) to make tests pass.
        - Ran unit tests: `test_download_mock.py` (Passed), `test_integration.py` (Passed).
        - Updated `HOUSEKEEPING.md` with the latest report.
