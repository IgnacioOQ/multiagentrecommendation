# TODOS: Recommender Systems & Learning Dynamics Study Plan
- status: active

> **Main Research Questions:**
> 1. How do recommender systems guide users into learning what they like?
> 2. How does non-stationarity affect learning dynamics?
> 3. Can recommender systems keep agents trapped in suboptimal equilibria when agents' reward functions are modulated?

---

## Phase 1: Foundation & Infrastructure ✅ (Partially Complete)
- status: active

### 1.1 Data Pipeline Setup
- status: active
- [x] Create `src/data/download.py` with `MovieLensPipeline` and `AmazonBeautyPipeline`
- [x] Create `src/data/process.py` for data transformation
- [x] Add unit tests (`tests/test_download_mock.py`)
- [x] Add integration tests (`tests/test_integration.py`)
- [x] Verify full pipeline execution: `python -m src.data.process` ✅ (100K+ MovieLens ratings, 198K+ Amazon reviews)

### 1.2 Models Training Pipeline
- status: active
- [x] Create `src/models/train_cf.py` for SVD collaborative filtering
- [x] Create `src/models/train_bandit.py` for LinUCB contextual bandits
- [x] Run and verify CF training on MovieLens data ✅
- [x] Run and verify bandit training on Amazon Beauty data ✅
- [x] Document baseline metrics (RMSE, MAE, mean rewards) ✅

#### Baseline Metrics (2026-01-22)
- status: active
| Model | Dataset | Metric | Value |
|-------|---------|--------|-------|
| SVD (CF) | MovieLens 100K | RMSE | 0.8729 ± 0.0033 |
| SVD (CF) | MovieLens 100K | MAE | 0.6712 ± 0.0033 |
| LinUCB (Custom) | Amazon Beauty (top 50 items) | Mean Reward (Replay) | 0.9040 |
| LinUCB (Custom) | Amazon Beauty (top 50 items) | Match Rate | 3.7% (427/11670) |

**Notes:**
- SVD model saved to `models/svd_model.pkl` (11MB)
- LinUCB policy saved to `models/bandit_policy.npz` (custom implementation)
- TF-IDF vectorizer saved to `models/tfidf_vectorizer.pkl`
- LinUCB uses TF-IDF features (100 dims) from review text as context
- Reward threshold for bandit: rating ≥ 4.0 → positive reward
- Custom LinUCB uses Sherman-Morrison O(d²) updates (no external dependency)

### 1.3 Simulation Pipeline Verification
- status: active
- [x] Existing `src/simulations.py` with `run_recommender_simulation()`
- [x] Existing `src/reward_modulators.py` with various modulator classes
- [x] Run sanity check simulation with default parameters ✅
- [x] Verify visualization outputs (Q-landscapes, reward maps) ✅

**Verified Visualization Functions (2026-01-22):**
- `plot_full_results()`: User and Recommender average reward heatmaps
- `plot_reward_statistics()`: Rolling mean, variance, cumulative reward
- `plot_initial_vs_final_qvalues()`: Q-value landscape evolution
- `plot_reward_distribution_analysis()`: Reward histograms and spatial distribution
- `plot_accept_reject_analysis()`: Accept/reject behavior over time
- `plot_learning_summary()`: Comprehensive 3x3 grid summary figure

---

## Phase 2: Theoretical Grounding
- status: active

### 2.1 Literature Review (Create Notes in `docs/`)
- status: active
- [ ] Review preference formation literature (how users learn preferences)
- [ ] Review non-stationarity in RL (drifting bandits, concept drift)
- [ ] Review reward shaping and intrinsic motivation literature
- [ ] Review exploration-exploitation trade-offs in recommender systems

### 2.2 Define Formal Model
- status: active
- [ ] Define agent utility function mathematically
- [ ] Define modulated reward function: `R_modulated(t) = f(R_true(t), modulator_state(t))`
- [ ] Define "suboptimality gap" metric
- [ ] Define "lock-in" or "local optima trapping" formally

### 2.3 Markov Chain Formalization (from MC_AGENT.md)
- status: active
- [ ] Define the state space formally:
  - User state: Q-values `Q_user(s,a)` for all (context, recommendation) pairs
  - Recommender state: Q-values or policy parameters
  - Modulator state: sensitivity level, history buffer, etc.
  - Combined state: `S_t = (Q_user, Q_rec, M_state, context)`
- [ ] Define transition dynamics:
  - Context transition: `context_{t+1} ~ P(·|context_t)` (environment drift)
  - User update: `Q_user_{t+1} = f(Q_user_t, action, modulated_reward)`
  - Recommender update: `Q_rec_{t+1} = g(Q_rec_t, action, rec_reward)`
  - Modulator update: `M_{t+1} = h(M_t, reward)`
- [ ] Identify sources of randomness:
  - Exploration (ε-greedy)
  - Environment stochasticity
  - Modulator dynamics (e.g., MoodSwings)

---

## Phase 3: Integration Layer
- status: active

### 3.1 Create Unified Experiment Interface
- status: active
- [ ] Create `src/experiments/config.py` with experiment configuration dataclasses
- [ ] Create `src/experiments/runner.py` that orchestrates:
  - Data loading (from `src/data/`)
  - Model training (from `src/models/`)
  - Simulation execution (from `src/simulations.py`)
- [ ] Support reproducibility (random seeds, logging)

### 3.2 Connect Data Pipeline to Simulation
- status: active
- [ ] Create adapter: MovieLens → Simulation Environment
  - Map movies to `n_recommendations` dimension
  - Map user contexts (genres, time) to `n_contexts` dimension
  - Use ratings as ground-truth reward landscape
- [ ] Create adapter: Amazon Beauty → Bandit Environment
  - Use TF-IDF features as context
  - Map products to arms

### 3.3 Define Experiment Protocols
- status: active
- [ ] **Protocol A: Stationary Learning Baseline**
  - No modulation, fixed environment
  - Measure: convergence time, final Q-landscape vs true landscape
- [ ] **Protocol B: Modulated Learning**
  - Apply reward modulators (ReceptorModulator, NoveltyModulator, HomeostaticModulator)
  - Measure: how modulation affects learned preferences
- [ ] **Protocol C: Non-Stationary Environment**
  - Use `shift_environment_right()` or similar
  - Measure: adaptation rate, tracking error

### 3.4 Markov Chain Analysis Infrastructure ✅
- status: active
Created `src/analysis/mc_analysis.py` with:

- [x] **State Tracking**
  ```python
  class MarkovChainAnalyzer:
      def snapshot_state(self) -> dict:
          """Capture full system state (user Q, rec Q, modulator state)"""
      
      def state_fingerprint(self) -> str:
          """Hashable representation for state comparison"""
      
      def compute_state_distance(self, state1, state2) -> float:
          """Measure distance between two states (e.g., Frobenius norm of Q-differences)"""
  ```

- [x] **Transition Analysis**
  ```python
  def estimate_transition_kernel(self, n_samples=1000) -> np.ndarray:
      """Estimate local transition probabilities via Monte Carlo"""
  
  def check_markov_property(self, n_tests=100) -> bool:
      """Verify transitions depend only on current state, not history"""
  ```

- [x] **Convergence Diagnostics** (mixing time placeholder)
  ```python
  def estimate_mixing_time(self, epsilon=0.01, n_chains=10) -> int:
      """Run parallel chains, measure when distributions merge"""
  
  def compute_spectral_gap(self) -> float:
      """Estimate spectral gap (larger = faster mixing)"""
  ```

- [x] **Absorption Analysis**
  ```python
  def estimate_absorption_probabilities(self, n_simulations=100) -> dict:
      """P(absorbing to global optimum) vs P(absorbing to local optimum)"""
  
  def mean_hitting_time(self, target_region, n_simulations=100) -> float:
      """Expected steps to reach target region of state space"""
  ```

---

## Phase 4: Core Experiments
- status: active

### 4.1 Experiment 1: Preference Formation (RQ1)
- status: active
**Question:** How do recommender systems guide users into learning what they like?

- [ ] Setup:
  - User agent starts with uniform/random Q-values
  - Recommender agent has learned (or optimal) policy
  - Track user's Q-landscape evolution over time
- [ ] Metrics:
  - Correlation between final user Q-landscape and environment true values
  - Time to "stabilize" preferences (mixing time proxy)
  - Diversity of explored options before convergence
- [ ] **MC Metrics:**
  - Track state trajectory: `S_0 → S_1 → ... → S_T`
  - Measure convergence to absorbing state
  - Verify Markov property holds
- [ ] Notebook: `notebooks/exp1_preference_formation.ipynb`

### 4.2 Experiment 2: Non-Stationarity Effects (RQ2)
- status: active
**Question:** How does non-stationarity affect learning dynamics?

- [ ] Setup:
  - Environment shifts periodically (`stationarity=False`)
  - Compare: user with/without modulation
  - Compare: different shift rates (slow vs fast)
- [ ] Metrics:
  - Tracking error: `||Q_learned(t) - Q_true(t)||`
  - Regret accumulation over time
  - "Staleness" of recommendations
- [ ] **MC Metrics:**
  - Analyze as time-inhomogeneous Markov chain
  - Measure "escape rate" from old equilibria when environment shifts
  - Compare mixing times under different drift rates
- [ ] Notebook: `notebooks/exp2_nonstationarity.ipynb`

### 4.3 Experiment 3: Suboptimal Lock-In (RQ3)
- status: active
**Question:** Can modulated reward functions trap agents in suboptima?

- [ ] Setup:
  - Environment has clear global optimum and local optimum
  - User agent uses modulated rewards (e.g., ReceptorModulator)
  - Compare: modulated vs non-modulated user
- [ ] Metrics:
  - Frequency of converging to global vs local optimum
  - Time spent in each region of state space
  - "Escape probability" from local optima
- [ ] **MC Metrics (Critical for RQ3):**
  - **Absorption probability**: `P(absorb to local | modulated)` vs `P(absorb to local | unmodulated)`
  - **Mean hitting time** to global optimum from initial state
  - **Spectral gap** comparison: modulation may reduce spectral gap (slower mixing = more trapping)
  - Identify new absorbing states created by modulation
- [ ] Hypothesis: Receptor downregulation may cause user to "tire" of high-reward options, preventing exploitation of global optimum
- [ ] Notebook: `notebooks/exp3_suboptimal_lockin.ipynb`

### 4.4 Experiment 4: Interaction Between Modulation Types
- status: active
- [ ] Compare different modulator classes:
  - `ReceptorModulator`: Tolerance/desensitization
  - `NoveltyModulator`: Exploration bonus decay
  - `HomeostaticModulator`: Setpoint regulation
  - `MoodSwings`: Stochastic drift in reward perception
- [ ] Measure: Which modulator leads to most/least "trapping"?
- [ ] **MC Metrics:**
  - Spectral gap per modulator type
  - Absorption probability distributions
  - State space coverage (ergodicity measure)
- [ ] Notebook: `notebooks/exp4_modulator_comparison.ipynb`

---

## Phase 5: Extended Analysis
- status: active

### 5.1 Visualization Suite
- status: active
- [ ] Create `src/visualization/` module with:
  - Q-landscape evolution animations
  - Reward trajectory plots
  - Accept/reject heatmaps over time
  - Modulator state trajectories
  - **NEW:** State space trajectory visualizations (2D/3D PCA projections)
  - **NEW:** Absorption basin visualizations

### 5.2 Statistical Analysis
- status: active
- [ ] Run multiple seeds (n=50+) for each experiment
- [ ] Compute confidence intervals
- [ ] Perform significance tests (paired t-tests, Mann-Whitney)
- [ ] **NEW:** Bootstrap estimates for absorption probabilities

### 5.3 Sensitivity Analysis
- status: active
- [ ] Vary modulator parameters (alpha, beta, etc.)
- [ ] Vary environment parameters (n_recommendations, n_contexts)
- [ ] Vary agent learning rates and exploration rates
- [ ] **NEW:** Sensitivity of spectral gap to parameters

### 5.4 Markov Chain Verification (NEW)
- status: active
- [ ] **Reproducibility Test**: Same seed → same trajectory
- [ ] **Markov Test**: Verify `P(X_{t+1} | X_t, X_{t-1}, ...) = P(X_{t+1} | X_t)`
- [ ] **Ergodicity Check**: Does the chain explore the full state space?
- [ ] **Stationarity Check**: Does a limiting distribution exist?

---

## Phase 6: Real Data Validation
- status: active

### 6.1 MovieLens Experiments
- status: active
- [ ] Initialize environment reward landscape from real ratings
- [ ] Simulate user learning with recommender guidance
- [ ] Compare: random recommender vs SVD-based recommender
- [ ] **MC Analysis:** Compare mixing times for different recommender policies

### 6.2 Amazon Beauty Experiments
- status: active
- [ ] Use contextual bandit in sequential recommendation
- [ ] Measure online learning performance
- [ ] Compare LinUCB vs random policy under modulation
- [ ] **MC Analysis:** Context-dependent absorption probabilities

---

## Phase 7: Documentation & Reporting
- status: active

### 7.1 Technical Documentation
- status: active
- [ ] Update `AGENTS.md` with new architecture
- [ ] Update `AI_AGENTS/MC_AGENT.md` with recommender-specific extensions
- [ ] Document all experiment configurations
- [ ] Add docstrings to new functions

### 7.2 Research Report
- status: active
- [ ] Write introduction with research questions
- [ ] Describe methodology (including MC framework)
- [ ] Present results with figures
- [ ] Discussion: implications for recommender system design
- [ ] Conclusion: when do recommenders help vs harm preference learning?

---

## Quick Reference: Key Files
- status: active

| File | Purpose |
|------|---------|
| `src/simulations.py` | Main simulation loop |
| `src/reward_modulators.py` | ReceptorModulator, NoveltyModulator, HomeostaticModulator, etc. |
| `src/agents/` | Q-Learning, DQN, PPO, Bandit agents |
| `src/environment.py` | ExogenousRewardEnvironment |
| `src/data/download.py` | MovieLens & Amazon data pipelines |
| `src/data/process.py` | Data transformation |
| `src/models/train_cf.py` | SVD collaborative filtering |
| `src/models/train_bandit.py` | LinUCB contextual bandit |
| `src/analysis/mc_analysis.py` | **NEW:** Markov Chain analysis tools |
| `AI_AGENTS/MC_AGENT.md` | Markov Chain Agent instructions |
| `AI_AGENTS/RECSYS_AGENT.md` | RecSys Agent instructions |

---

## Markov Chain Framework Summary
- status: active

### The System as a Markov Chain
- status: active

```
State Space: S_t = (Q_user, Q_rec, M_state, context)

Transition: S_t → S_{t+1}
  1. Recommender observes context → selects recommendation
  2. User observes (context, recommendation) → selects accept/reject
  3. Environment reveals reward
  4. Modulator transforms reward → modulated_reward
  5. User updates Q_user based on modulated_reward
  6. Recommender updates Q_rec based on accept/reject
  7. Modulator updates internal state
  8. Context transitions (possibly non-stationary)
```

### Key MC Questions for Each RQ
- status: active

| Research Question | MC Analysis |
|-------------------|-------------|
| RQ1: How do recommenders guide learning? | Track trajectory `S_0 → S_∞`, measure recommender's influence on absorption |
| RQ2: Effect of non-stationarity? | Time-inhomogeneous chain, measure adaptation via mixing time |
| RQ3: Suboptimal lock-in? | Compare absorption probabilities with/without modulation |

---

## Current Priority Queue
- status: active

1. **Immediate:** Verify data pipeline works end-to-end
2. **Next:** Run baseline simulation sanity check
3. **Then:** Create `src/analysis/mc_analysis.py` with basic state tracking
4. **After:** Design and run Experiment 1 (Preference Formation)
5. **Finally:** Introduce modulators and run Experiment 3 (Suboptimal Lock-In)

---

*Last Updated: 2026-01-21*
