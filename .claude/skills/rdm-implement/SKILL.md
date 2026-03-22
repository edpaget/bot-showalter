---
name: rdm-implement
description: Implement the next phase of an rdm roadmap
allowed-tools:
  - Read
  - Bash
  - Glob
  - Grep
  - Write
  - Edit
  - EnterPlanMode
  - ExitPlanMode
  - mcp__rdm__rdm_phase_list
  - mcp__rdm__rdm_phase_show
  - mcp__rdm__rdm_phase_update
  - mcp__rdm__rdm_task_create
---

Implement a phase from an rdm roadmap. `$ARGUMENTS` should be `<roadmap-slug> [phase-number]`.

## Principles

Read `docs/principles.md` before starting. It contains project conventions that should guide your work.
## Steps

1. **Parse arguments**: extract the roadmap slug and optional phase number from `$ARGUMENTS`.
2. **Find the phase**: if no phase number was given, use the `mcp__rdm__rdm_phase_list` tool with `project: "fbm"` and `roadmap: <slug>`, then pick the first `not-started` or `in-progress` phase.
3. **Read the phase**: use the `mcp__rdm__rdm_phase_show` tool with `project: "fbm"`, `roadmap: <slug>`, and `phase: <stem>` to get full context, steps, and acceptance criteria.
4. **Mark in-progress**: use the `mcp__rdm__rdm_phase_update` tool with `project: "fbm"`, `roadmap: <slug>`, `phase: <stem>`, and `status: "in-progress"`.
5. **Enter plan mode**: use the `EnterPlanMode` tool to switch into planning mode.
6. **Create an implementation plan** using the planning tool. The plan should:
   - Break the phase into concrete implementation steps based on the phase description and acceptance criteria
   - Include a final step: "Review changes with user and commit"
7. **Wait for user approval**: the user will review the plan and either accept or request changes. Do not proceed until the plan is accepted.
8. **Exit plan mode**: use the `ExitPlanMode` tool to switch back to execution mode.
9. **Execute the plan**: implement each step, following the plan and the phase's acceptance criteria.
10. **Review with user**: present a summary of the changes and ask the user to confirm they are ready to finalize.
11. **Finalize**: on user acceptance, commit the implementation changes with a `Done:` line in the commit message — the post-merge hook will mark the phase done and record the commit SHA.
    **Use the exact roadmap slug and phase stem from the rdm tools you used earlier — do NOT invent or paraphrase them:**
      ```
      Done: <roadmap-slug>/<phase-stem>
      ```
12. **Handle side-work**: if you discover bugs or unrelated improvements, create tasks using the `mcp__rdm__rdm_task_create` tool with `project: "fbm"`, a descriptive `slug`, `title`, and `body`.
