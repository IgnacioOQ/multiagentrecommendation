# Phase 2 Implementation: The Cloud Bridge
- status: active
- type: plan
- id: implementation.phase2
- owner: user
- priority: critical
- context_dependencies: {"master_plan": "../MASTER_PLAN.md", "conventions": "../../MD_CONVENTIONS.md"}
- last_checked: 2026-01-24T08:50:00+01:00
- blocked_by: [implementation.phase1]
<!-- content -->
This document details the "Cloud Bridge" implementation. The goal is to establish a secure, scalable communication channel between the Local Nexus (Phase 1) and the Cloud Agents (Phase 3) using the Google Cloud Ecosystem.

**Objective**:
1.  **Infrastructure**: Provision serverless compute and storage on GCP.
2.  **Connectivity**: Build a secure API Gateway for the local app to "phone home".
3.  **Synchronization**: Create pipelines to mirror local data to the cloud for heavy processing.

**Tech Stack**:
*   **Compute**: Google Cloud Run (Serverless Container).
*   **Database**: Google BigQuery (Warehousing) & Firestore (NoSQL Metadata).
*   **API**: Python FastAPI.
*   **Auth**: Firebase Authentication.
*   **Deployment**: Terraform / gcloud CLI (via Antigravity MCP).

## Module 1: Infrastructure Initialization (GCP)
- status: todo
- type: task
- id: implementation.phase2.infra
- estimate: 3d
- last_checked: 2026-01-24T08:50:00+01:00
<!-- content -->
Provision the necessary Google Cloud resources. We will favor "Infrastructure as Code" practices.

### Project Setup & API Enablement
- status: todo
- type: task
- id: implementation.phase2.infra.setup
- priority: high
<!-- content -->
*   **Action**: Create a new GCP Project (e.g., `intelligent-control-prod`).
*   **Enable APIs**:
    *   `run.googleapis.com` (Cloud Run)
    *   `artifactregistry.googleapis.com` (Docker Images)
    *   `bigquery.googleapis.com` (Data Warehouse)
    *   `firestore.googleapis.com` (App State)

### IaC & Deployment Workflow
- status: todo
- type: task
- id: implementation.phase2.infra.iac
- blocked_by: [implementation.phase2.infra.setup]
<!-- content -->
Define the infrastructure using Terraform or scriptable `gcloud` commands.
*   **Workflow**:
    1.  User prompts Antigravity to "Deploy Infrastructure".
    2.  Antigravity uses the terminal tool (or `gcloud` MCP) to execute the provisioning scripts.
    3.  Outputs (Service URLs, Bucket Names) are saved to `deployment_config.json`.

## Module 2: Authentication & Security
- status: todo
- type: task
- id: implementation.phase2.auth
- blocked_by: [implementation.phase2.infra]
- estimate: 1w
- last_checked: 2026-01-24T08:50:00+01:00
<!-- content -->
Secure the bridge. The Local App must authenticate before sending data.

### Identity Management (Firebase)
- status: todo
- type: task
- id: implementation.phase2.auth.firebase
<!-- content -->
*   **Setup**: Initialize a Firebase project linked to the GCP project.
*   **Client**: Integrate `firebase-admin` in the Cloud API and the JS/Python SDK in the Local App.
*   **Flow**:
    1.  Local User logs in.
    2.  Local App gets JWT Token.
    3.  API Gateway verifies JWT Token on every request.

### Service Security
- status: todo
- type: task
- id: implementation.phase2.auth.iam
<!-- content -->
*   **Service Accounts**: Create a specific Service Account for the Cloud Run instance.
*   **Permissions**: Grant strictly necessary roles (e.g., `roles/bigquery.dataEditor`, `roles/storage.objectCreator`). **Do not use Owner role.**

## Module 3: The API Gateway (Connector)
- status: todo
- type: task
- id: implementation.phase2.api
- blocked_by: [implementation.phase2.auth]
- estimate: 1w
- last_checked: 2026-01-24T08:50:00+01:00
<!-- content -->
Develop and deploy the central REST API.

### Service Skeleton (FastAPI)
- status: todo
- type: task
- id: implementation.phase2.api.dev
<!-- content -->
Create `src/cloud/main.py`.
*   **Endpoints**:
    *   `POST /v1/telemetry`: Accepts JSON payloads of user interactions.
    *   `POST /v1/agent/task`: Submits a complex task for the Cloud Agents.
    *   `GET /v1/agent/status/{task_id}`: Polling endpoint for long-running jobs.

### Containerization & Deploy
- status: todo
- type: task
- id: implementation.phase2.api.deploy
- blocked_by: [implementation.phase2.api.dev]
<!-- content -->
*   **Docker**: Create `Dockerfile` optimized for Python (multi-stage build).
*   **CI/CD**: Define a simple deployment script: `gcloud run deploy --source .`.

## Module 4: Data Synchronization Pipeline
- status: todo
- type: task
- id: implementation.phase2.pipeline
- blocked_by: [implementation.phase2.api]
- estimate: 1w
- last_checked: 2026-01-24T08:50:00+01:00
<!-- content -->
Mechanisms to move large datasets from Local DuckDB to Cloud BigQuery.

### Blob Storage Ingress
- status: todo
- type: task
- id: implementation.phase2.pipeline.gcs
<!-- content -->
For raw files (CSV/Excel) that are too large for JSON payloads.
*   **Mechanism**: Local App requests a Signed Upload URL from the API.
*   **Action**: Local App PUTs the file directly to a GCS Bucket (`raw-data-ingress`).

### Warehouse Sync (BigQuery)
- status: todo
- type: task
- id: implementation.phase2.pipeline.bigquery
- blocked_by: [implementation.phase2.pipeline.gcs]
<!-- content -->
*   **Schema Mapping**: Map DuckDB types to BigQuery types.
*   **Validation**: Check incoming schema against existing BigQuery schema to reject breaking changes (Schema Drift defense).
*   **Validation**: Check incoming schema against existing BigQuery schema to reject breaking changes (Schema Drift defense).
*   **ADK Compatibility**: Ensure the BigQuery dataset labels and descriptions are verbose. ADK's `BigQueryTool` uses these to understand how to query the data.
*   **Trigger**: When a file lands in GCS, a Cloud Event triggers a "Loader" function (or the API itself) to load the CSV into BigQuery.
