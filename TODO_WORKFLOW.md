---
status: active
type: plan
id: 'multiagentrecommendation.todo_workflow'
description: Cross-session task backlog; each task is self-contained and can be picked up by a coding agent with kb_mcp MCP tool access.
label: [planning, agent]
injection: excluded
volatility: evolving
owner: agent
last_checked: '2026-05-21'
---

# TODO Workflow

Cross-session task backlog. Tasks are added here when work started in a session cannot be completed immediately. Each task must be fully self-contained — a fresh agent should be able to pick it up using only the task body and the kb_mcp tools, with no additional context required.

This file is the per-repository instance of the `TODO_WORKFLOW_TEMPLATE.md` pattern. It lives at the root of the working repository alongside `worklog.jsonl` and is intentionally **not registered with kb_mcp** — agents access it via the regular filesystem `Read`/`Edit` tools, not via `knowledge_base_*` calls. To bootstrap a new repository, copy this template (`content/templates/TODO_WORKFLOW_TEMPLATE.md` in the knowledge base) to the repo root as `TODO_WORKFLOW.md` and fill in the `{{repo}}` and `{{YYYY-MM-DD}}` placeholders.

**Agent rules (picking up tasks):**
1. Read each task in full before starting. If its preconditions are unmet, skip it and note the blocker.
2. **Triage before committing.** If multiple tasks are open, scan them all and rank by value/difficulty ratio — do the cheapest high-value work first. Re-validate the author-set `difficulty` and `value` ratings against the current state of the repo before trusting them; conditions may have shifted since the task was written.
3. After completing a task, delete its entire block from this file (from the `---` divider above the `##` header through the `---` divider below the last line of the task body).
4. After completing one or more tasks, assess whether a `worklog.jsonl` entry is warranted (schema and append protocol below) — see also Phase 6 of `content/workflows/CODING_AGENT_MAIN_WORKFLOW.md`.
5. Confirm a task is still valid before executing; conditions may have changed since it was written.

**Adding tasks (session authors):**
- Copy the template below (without fences), fill in all fields, and insert it as a new `##` block above the Template section, preceded and followed by `---`.
- **Rate `difficulty` and `value`** (low / medium / high). Difficulty estimates effort and risk; value estimates impact on the repo, users, or future work. Pickup agents use this pair to triage the backlog.
- Be precise: include target file paths, specific tool calls, expected outcomes, and a verification step.
- Any `knowledge_base_update` call requires a current `content_hash` — capture it with a `knowledge_base_read` at execution time, not when writing the task.

## Worklog (`worklog.jsonl`) — Schema & Append Protocol

Each session that does non-trivial work appends one JSON object as a new line to `worklog.jsonl` at this repository's root. The file is plain JSONL — one JSON object per line, **oldest first** (chronological append order). It lives at root, outside any docs-discovery surface (kb_mcp, search indexers). There is no helper script; agents construct and append the JSON directly.

Bootstrap a new repo with an empty file: `touch worklog.jsonl`. The first entry's `session_id` is `1`.

### Schema (`schema_version: 1`)

```json
{
  "schema_version": 1,
  "entry_id":      "{{YYYY-MM-DD}}-s1",
  "date":          "{{YYYY-MM-DD}}",
  "session_id":    1,
  "summary":       "One-line task summary",
  "body_markdown": "- **Task:** ...\n- **Outcome:** ...\n- **Key decisions:** ...\n- **KB changes:** ...\n- **Follow-up:** ..."
}
```

| Field | Type | Notes |
|:--|:--|:--|
| `schema_version` | int | Currently `1`. Bump on breaking changes. |
| `entry_id` | string | Unique across the file. `YYYY-MM-DD-s{N}` when `session_id` is set; plain `YYYY-MM-DD` otherwise. Same-key collisions get `-b` / `-c` / `-d` suffixes. |
| `date` | string | ISO `YYYY-MM-DD`. |
| `session_id` | int \| null | Sequential session counter — last entry's `session_id` + 1. Use `null` if your repo doesn't track sessions. |
| `summary` | string | One-line heading — what the session accomplished. |
| `body_markdown` | string | Full narrative (Task / Outcome / Key decisions / KB changes / Follow-up) as one opaque markdown blob. The inner bullet structure is convention, not schema — pick whatever shape suits the entry. Newlines inside the string must be JSON-escaped as `\n`. |

### Append protocol

1. **Find the next `session_id`** — read the last line of `worklog.jsonl` (returns `1` if the file is empty):

    ```bash
    if [[ -s worklog.jsonl ]]; then
      tail -1 worklog.jsonl | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print((d.get('session_id') or 0)+1)"
    else
      echo 1
    fi
    ```

2. **Construct the entry as a single-line JSON object.** `json.dumps(entry, ensure_ascii=False)` handles all escaping. Verify uniqueness of `entry_id` against existing entries — if it collides, append `-b` / `-c` / `-d`.
3. **Append the line:**

    ```bash
    # Shell append (note: the JSON must be on ONE line)
    printf '%s\n' "$ENTRY_JSON" >> worklog.jsonl

    # Or Python (read-and-derive in one shot)
    python3 - <<'PY'
    import json
    entry = {
        "schema_version": 1,
        "entry_id":      "{{YYYY-MM-DD}}-s1",
        "date":          "{{YYYY-MM-DD}}",
        "session_id":    1,
        "summary":       "...",
        "body_markdown": "- **Task:** ...\n- **Outcome:** ...",
    }
    with open("worklog.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    PY
    ```

    Filesystem `Edit` works too — append the new JSON line after the last existing one.

4. **Skip the worklog append** for trivial one-line changes or purely exploratory sessions with no concrete output.

### Reading back

Render the latest N entries for reference / context loading:

```bash
tail -3 worklog.jsonl | python3 -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"
```

Or query with `jq`:

```bash
jq -r 'select(.session_id != null) | "\(.entry_id): \(.summary)"' worklog.jsonl | tail -10
```

---

## Reconcile stale AI_AGENTS agent docs

```yaml
status: todo
type: task
id: todo.reconcile_ai_agents_docs
description: Decide whether to rewrite or remove the AI_AGENTS agent docs whose body content describes a different project.
owner: agent
estimate: 30m
difficulty: low
value: medium
blocked_by: []
last_checked: '2026-05-21'
```

**Context:** During the 2026-05-21 MDDIA compliance migration it was found that `AI_AGENTS/LINEARIZE_AGENT.md` and `AI_AGENTS/MC_AGENT.md` describe a *network-epistemology* simulation (`BetaAgent`, credences, `networkx`, `model.py`, `vectorized_model.py`) — not this recommender-RL repository. Their MDDIA frontmatter is now compliant, but the body content is mismatched and references files that do not exist here. `AI_AGENTS/RECSYS_AGENT.md` is consistent with this project and needs no action.

**Preconditions:** none.

**Steps:**
1. Read `AI_AGENTS/LINEARIZE_AGENT.md` and `AI_AGENTS/MC_AGENT.md` in full.
2. With the repo owner, decide per file: (a) rewrite the body to match this recommender-RL project, or (b) delete the file if the specialized agent role is not needed.
3. Apply the decision. If rewriting, keep the existing MDDIA frontmatter; if deleting, remove the file and drop its entry from the `README.md` project tree.

**Verification:** `grep -rl 'BetaAgent\|networkx\|vectorized_model' AI_AGENTS/` returns no matches, and every file listed under `AI_AGENTS/` in `README.md` exists on disk.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md (from the `---` above the `##` header to the `---` below the last line).

---

## Fix broken `torch` install in the `rec-env` conda environment

```yaml
status: todo
type: task
id: todo.fix_torch_install
description: Repair the broken torch installation so the simulation modules import cleanly.
owner: agent
estimate: 15m
difficulty: medium
value: high
blocked_by: []
last_checked: '2026-05-21'
```

**Context:** The 2026-05-21 housekeeping run found the Phase 4 build smoke test (`python -c "import src.simulations, src.environment, src.reward_modulators"`) fails. The `rec-env` conda environment's `torch` package is corrupt: the `.dylib` files under `<env>/lib/python3.10/site-packages/torch/lib/` are symlinks pointing to a non-existent `libtorch_cpu.dylib` (and siblings). `src/reward_modulators.py` imports `torch`, so every simulation entrypoint is blocked. The unit/integration suite still passes because no test imports `torch`. Installed torch is 2.5.1; `requirements.txt` pins `torch==2.9.1`.

**Platform blocker (found during the 2026-05-21 reinstall attempt):** This is an Intel Mac (`x86_64`). PyTorch dropped x86 macOS wheels after **2.2.2** — so the pinned `torch==2.9.1` (and the currently-installed 2.5.1) have **no installable distribution on this hardware**. `pip install torch==2.9.1` fails with "No matching distribution found". The newest pip wheel for this platform is `torch-2.2.2-cp310-none-macosx_10_9_x86_64.whl`. Resolution requires a decision, not just a reinstall.

**Preconditions:** none.

**Steps (pick one path with the repo owner):**
1. **Downgrade path:** install the newest x86-macOS wheel — `/Users/ignacio/anaconda3/envs/rec-env/bin/python -m pip install --force-reinstall torch==2.2.2` — and update `requirements.txt` to `torch==2.2.2` so the pin matches reality. Then sanity-check that `src/reward_modulators.py` still works against the 2.2.x API.
2. **Hardware path:** keep the `torch==2.9.1` pin and rebuild the environment on an Apple Silicon (arm64) machine, where 2.9.1 wheels exist.
3. After whichever path: confirm the dylibs resolve — `ls -lL <env>/lib/python3.10/site-packages/torch/lib/libtorch_cpu.dylib` should show a real file, not a dangling link.

**Verification:** `python -c "import src.simulations, src.environment, src.reward_modulators"` exits 0 with no `ImportError`.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md (from the `---` above the `##` header to the `---` below the last line).

---

## Correct the `ReceptorModulator` test command path in README.md

```yaml
status: todo
type: task
id: todo.fix_readme_test_path
description: Update the README test command to reflect the file's location under tests/.
owner: agent
estimate: 5m
difficulty: low
value: low
blocked_by: []
last_checked: '2026-05-21'
```

**Context:** The 2026-05-21 housekeeping Phase 4 docs-freshness check found `README.md` § "Testing the `ReceptorModulator`" instructs `python test_receptor_modulator.py`, but the file lives at `tests/test_receptor_modulator.py`. Run from the repo root the documented command fails. `README.md` is an immutable core file — this fix requires explicit user approval before editing.

**Preconditions:** Explicit user instruction to edit `README.md` (core file — not editable during routine housekeeping).

**Steps:**
1. In `README.md`, change the command in the "Testing the `ReceptorModulator`" section from `python test_receptor_modulator.py` to `python tests/test_receptor_modulator.py`.

**Verification:** `grep -n "tests/test_receptor_modulator.py" README.md` returns the updated line; no bare `python test_receptor_modulator.py` remains.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md (from the `---` above the `##` header to the `---` below the last line).

---

## Task Template

Copy the block below (without the outer fences), fill in all fields, and insert it as a new `## [Task Title]` task block. Per-`##` metadata uses a fenced ` ```yaml ` block immediately after the heading (this file is a `plan` document, so the parser lifts these blocks into per-task metadata per Q8 of FRONTMATTER_MIGRATION_PLAN.md).

````markdown
## [Task Title]

```yaml
status: todo
type: task
id: todo.[short_id]
description: One-sentence description of what this task accomplishes.
owner: agent
estimate: Xm
difficulty: [low | medium | high]
value: [low | medium | high]
blocked_by: []
last_checked: '{{YYYY-MM-DD}}'
```

**Context:** Why this task exists and what triggered it. Include the KB path or repo file path it operates on.

**Preconditions:** Any state that must be true before starting (prior tasks complete, files present, etc.). Write `none` if there are none.

**Steps:**
1. (Include specific tool calls where possible, e.g., `knowledge_base_read(path="content/...", sections=["..."])`)
2. ...

**Verification:** How to confirm the task is complete (e.g., a grep that should return one match, a status field that should read `done`).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md (from the `---` above the `##` header to the `---` below the last line).
````
