# Master Plan: Recommender-Recommended RL Simulation
- status: active
- type: plan
- id: master_plan.recsys
- owner: ignacio
- priority: critical
- context_dependencies: {"manager": "../AI_AGENTS/MANAGER_AGENT.md", "conventions": "../MD_CONVENTIONS.md"}
<!-- content -->
This project simulates the dynamic interplay between a recommender system and a user, both modeled as reinforcement learning agents. It also uses real world data to inform the simulations. The ultimate goal of the project is to develop an **Offline Evaluation (or Offline Experimentation) platform for recommender systems**. In particular, it should help identify and explore how internal states and reward modulation can shape the user's learning and interaction over time.

## Core Concepts
- status: active
- type: context
- id: master_plan.recsys.concepts
<!-- content -->
The simulation is built around two key agents:
*   **Recommender Agent**: Learns to suggest items to the user. Goal: maximize accepted recommendations.
*   **User Agent (RecommendedAgent)**: Learns to accept or reject suggestions. Goal: maximize its own modulated reward.

## Key Components
- status: active
- type: context
- id: master_plan.recsys.components
<!-- content -->
*   **`agents.py`**: Q-Learning agents (Recommender & Recommended).
*   **`environment.py`**: 2D reward landscape (Context x Recommendation).
*   **`simulations.py`**: Core simulation runner.
*   **`reward_modulators.py`**: Psychological models (MoodSwings, ReceptorModulator, NoveltyModulator, HomeostaticModulator).

## Project Roadmap
- status: active
- type: plan
- id: master_plan.recsys.roadmap
<!-- content -->

### Phase 1: Foundation & Infrastructure
- status: done
- type: plan
- id: master_plan.recsys.phase1
<!-- content -->

#### 1.1 Data Pipeline Setup
- status: done
- type: task
- id: master_plan.recsys.phase1.data
<!-- content -->
- [x] Create `src/data/download.py` with `MovieLensPipeline` and `AmazonBeautyPipeline`
- [x] Create `src/data/process.py` for data transformation
- [x] Add unit tests (`tests/test_download_mock.py`)
- [x] Add integration tests (`tests/test_integration.py`)
- [x] Verify full pipeline execution: `python -m src.data.process` ✅ (100K+ MovieLens ratings, 198K+ Amazon reviews)

#### 1.2 Models Training Pipeline
- status: done
- type: task
- id: master_plan.recsys.phase1.models
<!-- content -->
- [x] Create `src/models/train_cf.py` for SVD collaborative filtering
- [x] Create `src/models/train_bandit.py` for LinUCB contextual bandits
- [x] Run and verify CF training on MovieLens data ✅
- [x] Run and verify bandit training on Amazon Beauty data ✅
- [x] Document baseline metrics (RMSE, MAE, mean rewards) ✅

#### 1.3 Simulation Pipeline Verification
- status: done
- type: task
- id: master_plan.recsys.phase1.sim
<!-- content -->
- [x] Existing `src/simulations.py` with `run_recommender_simulation()`
- [x] Existing `src/reward_modulators.py` with various modulator classes
- [x] Run sanity check simulation with default parameters ✅
- [x] Verify visualization outputs (Q-landscapes, reward maps) ✅

### Phase 2: Theoretical Grounding
- status: active
- type: plan
- id: master_plan.recsys.phase2
<!-- content -->

#### 2.1 Literature Review
- status: todo
- type: task
- id: master_plan.recsys.phase2.lit
<!-- content -->
- [ ] Review preference formation literature (how users learn preferences)
- [ ] Review non-stationarity in RL (drifting bandits, concept drift)
- [ ] Review reward shaping and intrinsic motivation literature
- [ ] Review exploration-exploitation trade-offs in recommender systems

#### 2.2 Define Formal Model
- status: todo
- type: task
- id: master_plan.recsys.phase2.model
<!-- content -->
- [ ] Define agent utility function mathematically
- [ ] Define modulated reward function: `R_modulated(t) = f(R_true(t), modulator_state(t))`
- [ ] Define "suboptimality gap" metric
- [ ] Define "lock-in" or "local optima trapping" formally

#### 2.3 Markov Chain Formalization
- status: todo
- type: task
- id: master_plan.recsys.phase2.mc
<!-- content -->
- [ ] Define the state space formally:
  - User state: Q-values `Q_user(s,a)`
  - Recommender state: Q-values or policy parameters
  - Modulator state: sensitivity level, history buffer
- [ ] Define transition dynamics
- [ ] Identify sources of randomness

### Phase 3: Integration Layer
- status: active
- type: plan
- id: master_plan.recsys.phase3
<!-- content -->

#### 3.1 Create Unified Experiment Interface
- status: done
- type: task
- id: master_plan.recsys.phase3.interface
<!-- content -->
- [x] Create `src/experiments/config.py` with experiment configuration dataclasses
- [x] Create `src/experiments/runner.py` that orchestrates data, models, and simulation
- [x] Support reproducibility (random seeds, logging)

#### 3.2 Connect Data Pipeline to Simulation
- status: done
- type: task
- id: master_plan.recsys.phase3.adapters
<!-- content -->
- [x] Create adapter: MovieLens → Simulation Environment
- [x] Create adapter: Amazon Beauty → Bandit Environment

#### 3.3 Define Experiment Protocols
- status: done
- type: task
- id: master_plan.recsys.phase3.protocols
<!-- content -->
- [x] **Protocol A: Stationary Learning Baseline**
- [x] **Protocol B: Modulated Learning**
- [x] **Protocol C: Non-Stationary Environment**

#### 3.4 Markov Chain Analysis Infrastructure
- status: done
- type: task
- id: master_plan.recsys.phase3.mc_infra
<!-- content -->
- [x] **State Tracking**: `src/analysis/mc_analysis.py` (Snapshot, Fingerprint)
- [x] **Transition Analysis**: Kernel estimation, Markov property check
- [x] **Convergence Diagnostics**: Mixing time, spectral gap
- [x] **Absorption Analysis**: Absorption probabilities, hitting time

### Phase 4: Core Experiments
- status: active
- type: plan
- id: master_plan.recsys.phase4
<!-- content -->

#### 4.1 Experiment 1: Preference Formation (RQ1)
- status: todo
- type: task
- id: master_plan.recsys.phase4.exp1
<!-- content -->
**Question:** How do recommender systems guide users into learning what they like?
- [ ] Setup: User starts with uniform Q-values
- [ ] Metrics: Correlation, stabilization time, diversity

#### 4.2 Experiment 2: Non-Stationarity Effects (RQ2)
- status: todo
- type: task
- id: master_plan.recsys.phase4.exp2
<!-- content -->
**Question:** How does non-stationarity affect learning dynamics?
- [ ] Setup: Shift environment (`stationarity=False`)
- [ ] Metrics: Tracking error, Regret, Staleness

#### 4.3 Experiment 3: Suboptimal Lock-In (RQ3)
- status: todo
- type: task
- id: master_plan.recsys.phase4.exp3
<!-- content -->
**Question:** Can modulated reward functions trap agents in suboptima?
- [ ] Setup: Global vs Local optimum env, compare with/without modulation
- [ ] Metrics: Absorption probability, Hitting time, Spectral gap

#### 4.4 Experiment 4: Interaction Between Modulation Types
- status: todo
- type: task
- id: master_plan.recsys.phase4.exp4
<!-- content -->
- [ ] Compare `ReceptorModulator`, `NoveltyModulator`, `HomeostaticModulator`, `MoodSwings`

### Phase 5: Extended Analysis
- status: active
- type: plan
- id: master_plan.recsys.phase5
<!-- content -->

#### 5.1 Visualization Suite
- status: todo
- type: task
- id: master_plan.recsys.phase5.viz
<!-- content -->
- [ ] Q-landscape evolution animations
- [ ] Reward trajectory plots
- [ ] State space trajectory visualizations

#### 5.2 Statistical Analysis
- status: todo
- type: task
- id: master_plan.recsys.phase5.stats
<!-- content -->
- [ ] Run multiple seeds (n=50+)
- [ ] Compute confidence intervals & significance tests

#### 5.3 Sensitivity Analysis
- status: todo
- type: task
- id: master_plan.recsys.phase5.sensitivity
<!-- content -->
- [ ] Vary modulator and environment parameters

#### 5.4 Markov Chain Verification
- status: todo
- type: task
- id: master_plan.recsys.phase5.verify
<!-- content -->
- [ ] Reproducibility, Markov, Ergodicity, Stationarity tests

### Phase 6: Real Data Validation
- status: active
- type: plan
- id: master_plan.recsys.phase6
<!-- content -->

#### 6.1 MovieLens Experiments
- status: todo
- type: task
- id: master_plan.recsys.phase6.ml
<!-- content -->
- [ ] Initialize env from real ratings
- [ ] Simulate user learning with Recommender guidance

#### 6.2 Amazon Beauty Experiments
- status: todo
- type: task
- id: master_plan.recsys.phase6.amazon
<!-- content -->
- [ ] Online learning performance with contextual bandit

### Phase 7: Documentation & Reporting
- status: active
- type: plan
- id: master_plan.recsys.phase7
<!-- content -->

#### 7.1 Technical Documentation
- status: todo
- type: task
- id: master_plan.recsys.phase7.tech
<!-- content -->
- [ ] Update `AGENTS.md`
- [ ] Document experiment configs

#### 7.2 Research Report
- status: todo
- type: task
- id: master_plan.recsys.phase7.report
<!-- content -->
- [ ] Introduction, Methodology, Results, Discussion, Conclusion

### Current Priority Queue
- status: active
- type: checklist
- id: master_plan.recsys.priority
<!-- content -->
1. **Immediate:** Design and run Experiment 1 (Preference Formation)
2. **Next:** Introduce modulators and run Experiment 3 (Suboptimal Lock-In)
3. **Then:** Complete Phase 2 Formal Modeling
