# AGENTS.md
- status: active
- type: guideline
- context_dependencies: {"conventions": "MD_CONVENTIONS.md", "log": "AGENTS_LOG.md", "housekeeping": "HOUSEKEEPING.md"}
<!-- content -->

## SHORT ADVICE
- status: active
<!-- content -->
- The whole trick is providing the AI Assistants with context, and this is done using the *.md files (AGENTS.md, AGENTS_LOG.md, and the AI_AGENTS folder)
- Learn how to work the Github, explained below.
- Keep logs of changes in AGENTS_LOG.md
- Make sure to execute the HOUSEKEEPING.md protocol often.
- Always ask several forms of verification, so because the self-loop of the chain of thought improves performance.
- Impose restrictions and constraints explicitly in the context.

## HUMAN-ASSISTANT WORKFLOW
- status: active
<!-- content -->
1. Open the assistant and load the ai-agents-branch into their local repositories. Do this by commanding them to first of all read the AGENTS.md file.
2. Work on the ASSISTANT, making requests, modifying code, etc.
3. IMPORTANT: GIT MECHANISM
    3.1. This is basically solved in Antigravity, but Jules (and maybe Claude) push the changes into a newly generated branch. In my case, this is `jules-sync-main-v1-15491954756027628005`. **This is different from the `ai-agents-branch`!!**
    3.2. So what you need to do is merge the newly generated branch and the `ai-agents-branch` often. Usually in the direction from `jules-sync-main-v1-15491954756027628005` to `ai-agents-branch`. I do this by:
        3.2.1. Going to pull requests.
        3.2.2. New Pull request
        3.2.3. Base: `ai-agents-branch`, Compare: `jules-sync-main-v1-15491954756027628005` (arrow in the right direction).
        3.2.4. Follow through. It should allow to merge and there should not be incompatibilities. If there are incompatibilities, you can delete the `ai-agents-branch` and create a new one cloning the `jules-sync-main-v1-15491954756027628005` one. After deleting `ai-agents-branch`, go to the `jules-sync-main-v1-15491954756027628005` branch, look at the dropdown bar with the branches (not the link), and create a new copy.
4. It is very useful to use specialized agents for different sectors of the code. 
5. Enjoy!

## WORKFLOW & TOOLING
- status: active
<!-- content -->
*   **Documentation Logs (`AGENTS_LOG.md`):**
    *   **Rule:** Every agent that performs a significant intervention or modifies the codebase **MUST** update the `AGENTS_LOG.md` file.
    *   **Action:** Append a new entry under the "Intervention History" section summarizing the task, the changes made, and the date.

## DEVELOPMENT RULES & CONSTRAINTS
- status: active
<!-- content -->
1.  **Immutable Core Files:** Do not modify 
    *   If you need to change the logic of an agent or the model, you must create a **new version** (e.g., a subclass or a new file) rather than modifying the existing classes in place.
2.  **Consistency:** Ensure any modifications or new additions remain as consistent as possible with the logic and structure of the `main` branch.
3.  **Coding Conventions:** Always keep the coding conventions pristine.

## CONTEXT FINE-TUNING
- status: active
<!-- content -->
You cannot "fine-tune" an AI agent (change its underlying neural network weights) with files in this repository. **However**, you **CAN** achieve a similar result using **Context**.

**How it works (The "Context" Approach):**
If you add textbooks or guides to the repository (preferably as Markdown `.md` or text files), agents can read them. You should then update the relevant agent instructions (e.g., `AI_AGENTS/LINEARIZE_AGENT.md`) to include a directive like:

> "Before implementing changes, read `docs/linearization_textbook.md` and `docs/jax_guide.md`. Use the specific techniques described in Chapter 4 for sparse matrix operations."

**Why this is effective:**
1.  **Specific Knowledge:** Adding a specific textbook helps if you want a *specific style* of implementation (e.g., using `jax.lax.scan` vs `vmap` in a particular way).
2.  **Domain Techniques:** If the textbook contains specific math shortcuts for your network types, providing the text allows the agent to apply those exact formulas instead of generic ones.

**Recommendation:**
If you want to teach an agent a new language (like JAX) or technique:
1.  Add the relevant chapters as **text/markdown** files.
2.  Update the agent's instruction file (e.g., `AI_AGENTS/LINEARIZE_AGENT.md`) to reference them.
3.  Ask the agent to "Refactor the code using the techniques in [File X]".

## LOCAL PROJECT DESCRIPTION
- status: active
<!-- content -->

### Project Overview
- status: active
<!-- content -->
This project simulates a **multi-agent reinforcement learning system** modeling the dynamic interaction between a **recommender system** and a **user**, both implemented as Q-learning agents. The core research focus is exploring how **internal reward modulation** (inspired by biological and psychological models) shapes agent learning, decision-making, and long-term behavior in recommendation systems.

**The Two Agents:**
- **RecommenderAgent**: Learns to suggest items that maximize user acceptance. Receives +1 reward for accepted recommendations, -1 for rejections.
- **RecommendedAgent (User)**: Learns to accept/reject recommendations to maximize personal reward from the environment. Can experience modulated rewards based on internal states.

**Research Objectives:**
- Investigate how reward modulation mechanisms (mood, tolerance, novelty-seeking, homeostasis) affect user learning
- Explore multi-agent dynamics in human-in-the-loop recommendation systems
- Model biologically-inspired internal states and their impact on decision-making

### Setup & Testing
- status: active
<!-- content -->
**Installation:**
```bash
pip install -r requirements.txt
```

**Running Tests:**
```bash
python -m unittest discover tests
# OR
python tests/test_receptor_modulator.py
```

**Running Simulations:**
Use the provided Jupyter notebooks in `notebooks/` for interactive experimentation:
- `testing_homeostasis.ipynb` - Tests homeostatic reward modulators
- `testing_peaks.ipynb` - Tests reward landscapes with multiple peaks
- `testing_rows.ipynb` - Tests row-based reward landscapes

**Basic Usage:**
```python
from src.simulations import run_recommender_simulation
from src.agents.q_learning import RecommenderAgent, RecommendedAgent
from src.environment import ExogenousRewardEnvironment
from src.reward_modulators import ReceptorModulator

results = run_recommender_simulation(
    recommender_agent_class=RecommenderAgent,
    recommended_agent_class=RecommendedAgent,
    environment_class=ExogenousRewardEnvironment,
    modulator_class=ReceptorModulator,
    n_steps=10000,
    modulated=True
)
```

### Key Architecture & Logic
- status: active
<!-- content -->

#### 1. Core Logic
- status: active
<!-- content -->

**Agent System (`src/agents/`):**
- `BaseQLearningAgent`: Foundation class implementing Q-learning with multiple exploration strategies (ε-greedy, UCB, softmax)
- `RecommenderAgent`: Chooses recommendations (actions) based on context states to maximize acceptance rate
- `RecommendedAgent`: Chooses accept/reject actions based on (context, recommendation) state pairs to maximize personal reward

**Environment (`src/environment.py`):**
- `ExogenousRewardEnvironment`: Creates a 2D reward landscape
  - X-axis: Contexts (user states, typically 50 discrete values)
  - Y-axis: Recommendations (items, typically 20 discrete values)
  - Each cell contains the true reward value for accepting that recommendation in that context
  - Supports Gaussian peaks (global max, local max) to create complex reward landscapes
  - Dynamic context transitions: agent moves through context space over time
  - Optional non-stationarity: reward landscape can shift during simulation

**Simulation Loop (`src/simulations.py`):**
1. Get current context from environment
2. Recommender chooses a recommendation
3. User decides to accept/reject
4. If accepted: user receives environment reward (potentially modulated)
5. If rejected: user receives 0 reward
6. Recommender receives +1 (accept) or -1 (reject)
7. Both agents update their Q-tables
8. Environment transitions to new context
9. Repeat for n_steps

**Reward Modulators (`src/reward_modulators.py`):**

The project's most unique contribution. These modulators alter the user's *perceived* reward based on internal states:

- **MoodSwings**: Adds time-varying mood offset to rewards (models emotional fluctuations)
- **ReceptorModulator**: Implements tolerance/sensitization
  - Sensitivity decreases with repeated high rewards (downregulation)
  - Sensitivity recovers during low reward periods (upregulation)
  - Models biological receptor dynamics (e.g., dopamine tolerance)
- **NoveltyModulator**: Adds bonus for infrequent (context, recommendation) pairs (models curiosity/exploration drive)
- **HomeostaticModulator**: Q-learning-based controller that modulates rewards to maintain homeostatic setpoint
- **TD_DHR / DQN_DHR**: Advanced homeostatic controllers using tabular Q-learning or deep Q-networks
  - Learn to select modulation signals from reward history to minimize deviation from setpoint
  - **_D variants**: Dynamic setpoint (allostasis) - setpoint adapts over time
  - **_E variants**: Expanded action space - can generate novel modulation signals beyond history

#### 2. Dependencies (`src/imports.py`)
- status: active
<!-- content -->
*   Centralizes imports for `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`, `torch`, `tqdm`, and `collections`.
*   Provides common utilities used across all modules.

### Key Files and Directories
- status: active
<!-- content -->

#### Directory Structure
- status: active
<!-- content -->
*   `src/`: Contains the core Python modules.
    *   `agents/`: Q-Learning Agent classes (BaseQLearningAgent, RecommenderAgent, RecommendedAgent in `q_learning.py`).
    *   `utils/`: Utility functions (e.g., `math_ops.py`).
    *   `environment.py`: ExogenousRewardEnvironment class.
    *   `simulations.py`: Main simulation runner.
    *   `reward_modulators.py`: Reward modulation classes (MoodSwings, ReceptorModulator, etc.).
    *   `plotting_utils.py`: Plotting functions.
    *   `stationarity_analysis.py`: Statistical tests for time series stationarity.
    *   `imports.py`: Common imports shared across modules.
*   `tests/`: Contains unit tests.
    *   `test_receptor_modulator.py`: Tests for reward modulator behavior.
    *   `test_integration.py`: Integration tests.
    *   `test_download_mock.py`: Mock download tests.
*   `notebooks/`: Contains Jupyter notebooks for visualization and experimentation.
    *   `testing_peaks.ipynb`: Peak analysis experiments.
    *   `testing_homeostasis.ipynb`: Homeostasis behavior tests.
    *   `testing_rows.ipynb`: Row-wise landscape experiments.
*   `AI_AGENTS/`: Contains specialized agent instruction files.
    *   `MC_AGENT.md`: Markov Chain Analysis Agent instructions.

#### File Dependencies & Logic
- status: active
<!-- content -->
`src/simulations.py` depends on `src/agents/`, `src/environment.py`, and `src/reward_modulators.py`.
All source files import from `src/imports.py` for common dependencies.

**Implementation:**
The Q-Learning agents process states using tabular Q-values. Training uses epsilon-greedy exploration with configurable decay.

**User Interface:**
The interface is provided via Jupyter notebooks in `notebooks/`, allowing visualization of agent behavior and reward landscapes.

**Testing & Verification:**
*   **`tests/test_receptor_modulator.py`**: Tests reward modulator behavior under different reward sequences.
