# AGENTS.md

## SHORT ADVICE
- The whole trick is providing the AI Assistants with context, and this is done using the *.md files (AGENTS.md, AGENTS_LOG.md, and the AI_AGENTS folder)
- Learn how to work the Github, explained below.
- Keep logs of changes in AGENTS_LOG.md
- Always ask several forms of verification, so because the self-loop of the chain of thought improves performance.
- Impose restrictions and constraints explicitly in the context.

## HUMAN-ASSISTANT WORKFLOW
1. Open the assistant and load the ai-agents-branch into their local repositories. Do this by commanding them to first of all read the AGENTS.md file.
2. Work on the ASSISTANT, making requests, modifying code, etc.
3. IMPORTANT: GIT MECHANISM
    3.1. Jules (and maybe Claude) push the changes into a newly generated branch. In my case, this is `jules-sync-main-v1-15491954756027628005`. **This is different from the `ai-agents-branch`!!**
    3.2. So what you need to do is merge the newly generated branch and the `ai-agents-branch` often. Usually in the direction from `jules-sync-main-v1-15491954756027628005` to `ai-agents-branch`. I do this by:
        3.2.1. Going to pull requests.
        3.2.2. New Pull request
        3.2.3. Base: `ai-agents-branch`, Compare: `jules-sync-main-v1-15491954756027628005` (arrow in the right direction).
        3.2.4. Follow through. It should allow to merge and there should not be incompatibilities. If there are incompatibilities, you can delete the `ai-agents-branch` and create a new one cloning the `jules-sync-main-v1-15491954756027628005` one. After deleting `ai-agents-branch`, go to the `jules-sync-main-v1-15491954756027628005` branch, look at the dropdown bar with the branches (not the link), and create a new copy.
4. Enjoy!

## WORKFLOW & TOOLING
*   **PostToolUse Hook (Code Formatting):**
    *   **Context:** A "hook" is configured to run automatically after specific events.
    *   **The Event:** "PostToolUse" triggers immediately after an agent uses a tool to modify a file (e.g., writing code or applying an edit).
    *   **The Action:** The system automatically runs a code formatter (like `black` for Python) on the modified file.
    *   **Implication for Agents:** You do not need to manually run a formatter. The system handles it. However, be aware that the file content might slightly change (whitespace, indentation) immediately after you write to it.

*   **Jupyter Notebooks (`.ipynb`):**
    *   **Rule:** Do not attempt to read or edit `.ipynb` files directly with text editing tools. They are JSON structures and easy to corrupt.
    *   **Action:** If you need to verify or modify logic in a notebook, ask the user to export it to a Python script, or create a new Python script to reproduce the logic.
    *   **Exception:** You may *run* notebooks if the environment supports it (e.g., via `nbconvert` to execute headless), but avoid editing the source.

*   **Documentation Logs (`AGENTS_LOG.md`):**
    *   **Rule:** Every agent that performs a significant intervention or modifies the codebase **MUST** update the `AGENTS_LOG.md` file.
    *   **Action:** Append a new entry under the "Intervention History" section summarizing the task, the changes made, and the date.

## DEVELOPMENT RULES & CONSTRAINTS
1.  **Immutable Core Files:** Do not modify `agents.py`, `model.py`, or `simulation_functions.py` (if they exist in this context, otherwise apply to core logic files like `functions.py`).
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
**Homeostatic Reinforcement Learning**: This project simulates the interaction between a Recommender Agent and a Recommended Agent (User) within a 2D environment of contexts and recommendations. The core focus is on **homeostatic modulation**, where the User Agent's perception of reward is modified by internal states (e.g., satiety, receptor downregulation, novelty seeking, or maintenance of a specific setpoint). This allows for modeling complex, non-stationary user behaviors and exploring "Allostatic" control strategies (dynamic setpoints).

### Setup & Testing
*   **Installation:** `pip install -r requirements.txt`
*   **Testing:**
    *   Unit tests should be placed in `tests/`.
    *   Integration verification can be performed by running the provided Jupyter notebooks (`testing_homeostasis.ipynb`, `testing_peaks.ipynb`, etc.).
    *   Sanity checks for specific components (like `ReceptorModulator`) are available as scripts (e.g., `test_receptor_modulator.py`).

### Key Architecture & Logic

#### 1. Core Logic
*   **`agents.py`**: Defines `BaseQLearningAgent` (standard Q-learning) and its subclasses `RecommenderAgent` (selects items) and `RecommendedAgent` (accepts/rejects items).
*   **`environment.py`**: Defines `ExogenousRewardEnvironment`, a 2D grid of rewards with gaussian peaks representing user preferences across contexts. Supports non-stationarity (shifting landscapes).
*   **`reward_modulators.py`**: The heart of the homeostatic logic. Contains:
    *   `ReceptorModulator`: Simulates receptor desensitization/downregulation.
    *   `HomeostaticModulator`: Maintains a reward setpoint via lagged modulation.
    *   `TD_DHR` / `DQN_DHR`: Uses Temporal Difference learning or Deep Q-Networks to learn the optimal modulation strategy to maintain homeostasis (Deep Homeostatic Reinforcement).
    *   `TD_DHR_D` / `DQN_DHR_D`: Adds Allostasis (Dynamic Setpoints).
*   **`simulations.py`**: Orchestrates the interaction loop between agents, environment, and modulators.

#### 2. Dependencies (`imports.py`)
*   Centralizes imports for `pandas`, `yfinance`, `matplotlib`, `tqdm`, `requests`, `bs4`, `torch`, `numpy`, and `scipy`.

### Key Files and Directories
*   `agents.py`: Q-learning agent implementations.
*   `environment.py`: Environment simulation.
*   `reward_modulators.py`: Logic for homeostatic and allostatic reward adjustment.
*   `simulations.py`: Simulation runners.
*   `tests/`: Directory for unit tests.
*   `testing_*.ipynb`: Notebooks for visual verification and experiments.
