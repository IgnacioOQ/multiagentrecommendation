# Phase 3 Implementation: The Cloud Agents
- status: todo
- type: plan
- id: implementation.phase3
- owner: user
- priority: critical
- estimate: 6w
- blocked_by: [implementation.phase2]
- context_dependencies: {"master_plan": "../MASTER_PLAN.md", "conventions": "../../MD_CONVENTIONS.md"}
<!-- content -->
This document details the implementation of the "Brain" of the Intelligent Control SaaS: a multi-agent system built using the **Google Agent Development Kit (ADK)**.

**Objective**: Deploy a robust, observable, and scalable agent ecosystem handling:
1.  **Analysis**: Python-based data science and visualization.
2.  **Control**: RL/Control-theory optimization using a custom algorithm repository.
3.  **Orchestration**: Intelligent routing and state management.

**Tech Stack**:
*   **Framework**: Google ADK (Python SDK).
*   **Model**: Gemini 1.5 Pro (via Vertex AI).
*   **Runtime**: Cloud Run (Containerized Agents).
*   **Evaluation**: Vertex AI Gen AI Evaluation Service.

## Architecture: The ADK Ecosystem
- status: todo
- type: plan
- id: implementation.phase3.arch
- estimate: 1w
<!-- content -->
We will leverage ADK's pattern for composable agents. The system will consist of a top-level **Coordinator Agent** and two specialized worker agents.

### The Coordinator Pattern
- status: todo
- type: protocol
- id: implementation.phase3.arch.coordinator
<!-- content -->
Instead of a monolithic chain, we use a central `LlmAgent` acting as a router.
*   **Input**: Natural language user queries + State Context (from Phase 2).
*   **Decision**: Uses a `classify_intent` tool or few-shot prompting to decide:
    *   `ANALYSIS_REQUIRED` -> Delegate to Analyst Agent.
    *   `CONTROL_REQUIRED` -> Delegate to Controller Agent.
    *   `AMBIGUOUS` -> Ask clarifying questions.
*   **Output**: Aggregates responses from workers and formats the final answer for the user.

## Module 1: The Analyst Agent (Data Scientist)
- status: todo
- type: task
- id: implementation.phase3.analyst
- blocked_by: [implementation.phase3.arch]
- estimate: 2w
<!-- content -->
**Role**: "Why is this happening?"
**Tools**: Code Execution, Data Visualization.

### Data Science Tool Repository
- status: todo
- type: task
- id: implementation.phase3.analyst.repo
<!-- content -->
We will build a dedicated Python library (`src/lib_analysis`) that the agent learns to use.
*   **Structure**:
    ```python
    /src/lib_analysis
       /visualize.py   # High-level plot wrappers (plot_time_series, plot_distribution)
       /stats.py       # Hypothesis testing (anova, t_test)
       /clean.py       # Auto-cleaning utilities
    ```
*   **Integration**:
    *   Expose these functions as **ADK Tools**.
    *   Use type hints and docstrings heavily, as ADK uses these for tool definition verification.

### Code Execution Sandbox
- status: todo
- type: task
- id: implementation.phase3.analyst.sandbox
<!-- content -->
*   **Mechanism**: The agent writes code that imports `lib_analysis`.
*   **Security**: Use ADK's `CodeExecutionTool` configured with a restricted environment (or E2B integration if ADK native support is insufficient).
*   **Output Handling**: Capture `stdout` (text) and generated artifacts (PNG/JSON) to pass back to the Coordinator.

## Module 2: The Controller Agent (Optimizer)
- status: todo
- type: task
- id: implementation.phase3.controller
- blocked_by: [implementation.phase3.analyst]
- estimate: 2w
<!-- content -->
**Role**: "Optimize for X."
**Tools**: Optimization Algorithms, Simulation.

### Control Algorithms Integration
- status: todo
- type: task
- id: implementation.phase3.controller.integration
<!-- content -->
Integrate the external repository [control_algorithms](https://github.com/IgnacioOQ/control_algorithms).
*   **Step 1**: Submodule or Package integration of the user's repository.
*   **Step 2**: Create an **ADK Tool Wrapper** (`src/tools/control_tools.py`) that exposes key algorithms as callable functions:
    *   `run_mpc_optimization(state_vector, constraints)`
    *   `solve_newsvendor(demand_dist, costs)`
    *   `simulate_scenario(initial_state, horizon)`
*   **Step 3**: Define the "State Schema". The Agent must know how to map the raw telemetry (from BigQuery/Phase 2) into the inputs required by these algorithms.

## Module 3: Agent Development & Ops (ADK)
- status: todo
- type: task
- id: implementation.phase3.ops
- blocked_by: [implementation.phase3.controller]
- estimate: 1w
<!-- content -->
Establish the lifecycle for developing and improving these agents.

### Evaluation Pipeline (GenAI Eval)
- status: todo
- type: task
- id: implementation.phase3.ops.eval
<!-- content -->
Use Google's Gen AI Evaluation Service to move beyond "vibes-based" testing.
*   **Trajectory Evaluation**: check if the Analyst Agent *actually* used the `visualize.py` tool or if it tried to hallucinate a plot.
*   **Golden Datasets**: Create a set of (Query, Expected_Tool_Call, Expected_Outcome) tuples.
*   **CI/CD**: Run `adk eval` as part of the deployment pipeline.

### Deployment (Vertex AI)
- status: todo
- type: task
- id: implementation.phase3.ops.deploy
<!-- content -->
*   **Containerize**: Wrap the ADK agent server in a Docker container.
*   **Deploy**: Push to Cloud Run.
*   **Expose**: Connect the Cloud Run endpoint to the API Gateway created in Phase 2.
