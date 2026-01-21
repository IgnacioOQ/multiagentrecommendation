# AGENTS.md

## SHORT ADVICE
- The whole trick is providing the AI Assistants with context, and this is done using the *.md files (AGENTS.md, AGENTS_LOG.md, and the AI_AGENTS folder)
- Learn how to work the Github, explained below.
- Keep logs of changes in AGENTS_LOG.md
- Make sure to execute the HOUSEKEEPING.md protocol often.
- Always ask several forms of verification, so because the self-loop of the chain of thought improves performance.
- Impose restrictions and constraints explicitly in the context.

## HUMAN-ASSISTANT WORKFLOW
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

*   **Documentation Logs (`AGENTS_LOG.md`):**
    *   **Rule:** Every agent that performs a significant intervention or modifies the codebase **MUST** update the `AGENTS_LOG.md` file.
    *   **Action:** Append a new entry under the "Intervention History" section summarizing the task, the changes made, and the date.

## DEVELOPMENT RULES & CONSTRAINTS
1.  **Immutable Core Files:** Do not modify 
    *   If you need to change the logic of an agent or the model, you must create a **new version** (e.g., a subclass or a new file) rather than modifying the existing classes in place.
2.  **Consistency:** Ensure any modifications or new additions remain as consistent as possible with the logic and structure of the `main` branch.
3.  **Coding Conventions:** Always keep the coding conventions pristine.

## CONTEXT FINE-TUNING
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

### Project Overview

### Setup & Testing
*   **Install Dependencies:**
*   **Run Backend:** 
*   **Run Frontend:** 
*   **Run Tests:** 

### Key Architecture & Logic

#### 1. Architecture (Monorepo)
*   **Backend (`backend/`)**: Python/FastAPI application handling all simulation logic, agent training, and state management.
*   **Frontend (`frontend/`)**: React/Vite application for the user interface, communicating with backend via REST API.

#### 2. Agents
*   **`backend/environment.py`**: The environment logic (p generation, reward calculation).
*   **`backend/agents.py`**: The Deep Q-Learning (DQN) agent implementation using PyTorch.

#### 3. Simulation Loop (`backend/simulation.py`)
*   **Step:**
    1.  Environment generates p.
    2.  Agents observe p and output actions (Recommend/Not Recommend).
    3.  Human (via React Interface) observes Agent actions and selects one.
    4.  Environment calculates outcome (Coin flip) and rewards.
    5.  Agents update their replay buffers and perform a training step.

### Key Files and Directories

#### Directory Structure
*   `src/`: Contains the core Python modules.
    *   `agents.py`: Q-Learning Agent classes (BaseQLearningAgent, RecommenderAgent, RecommendedAgent).
    *   `environment.py`: ExogenousRewardEnvironment class.
    *   `simulations.py`: Main simulation runner.
    *   `reward_modulators.py`: Reward modulation classes (MoodSwings, ReceptorModulator, etc.).
    *   `utils.py`: Plotting and utility functions.
    *   `stationarity_analysis.py`: Statistical tests for time series stationarity.
    *   `imports.py`: Common imports shared across modules.
*   `tests/`: Contains unit tests.
    *   `test_receptor_modulator.py`: Tests for reward modulator behavior.
*   `notebooks/`: Contains Jupyter notebooks for visualization and experimentation.
    *   `testing_peaks.ipynb`: Peak analysis experiments.
    *   `testing_homeostasis.ipynb`: Homeostasis behavior tests.
    *   `testing_rows.ipynb`: Row-wise landscape experiments.
*   `AI_AGENTS/`: Contains specialized agent instruction files.
    *   `MC_AGENT.md`: Markov Chain Analysis Agent instructions.

#### File Dependencies & Logic
`src/simulations.py` depends on `src/agents.py`, `src/environment.py`, and `src/reward_modulators.py`.
All source files import from `src/imports.py` for common dependencies.

**Implementation:**
The Q-Learning agents process states using tabular Q-values. Training uses epsilon-greedy exploration with configurable decay.

**User Interface:**
The interface is provided via Jupyter notebooks in `notebooks/`, allowing visualization of agent behavior and reward landscapes.

**Testing & Verification:**
*   **`tests/test_receptor_modulator.py`**: Tests reward modulator behavior under different reward sequences.
