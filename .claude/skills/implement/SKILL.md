---
name: implement
description: Implement a phase from an existing roadmap. Reads the roadmap, enters plan mode, and guides the full implement-and-merge cycle. Use when the user asks to "implement phase N of X" or "start working on X".
argument-hint: <roadmap> [phase N]
---

# Implement Skill

Implement a specific phase from an existing roadmap document.

## Behavior

1. Parse `$ARGUMENTS` to identify the roadmap name (kebab-case, matching a file in `docs/plans/`) and the phase number. If the phase number is omitted, read the roadmap's Status table and pick the first phase that is `not started`.
2. Read `docs/plans/<roadmap>.md` and extract the target phase's steps and acceptance criteria.
3. Verify the phase is not already marked `done` or `in progress` in the Status table. If it is, inform the user and stop.
4. Check if the phase has any blockers (other roadmap phases listed as dependencies in the Ordering section). If blocked, inform the user and stop.
5. Update the roadmap's Status table to mark the phase `in progress`.
6. Enter plan mode — explore the codebase, design the implementation approach against the roadmap's steps and acceptance criteria, and present the plan for user approval.
7. After plan-mode approval, create a worktree via `EnterWorktree` (named `roadmap/<topic>/phase-<N>`).
8. Implement following the project's Implementation Discipline (TDD, run tests after each step, verify all acceptance criteria before committing).
9. After the final commit, update the roadmap's Status table to mark the phase `done (<date>)`.

## Argument parsing

- `$ARGUMENTS` may look like: `draft-board-export phase 3`, `draft-board-export 3`, `tier-generator`, `tier-generator phase 1`
- The roadmap name is everything before an optional `phase` keyword or trailing number.
- If no phase number is given, auto-select the next `not started` phase.

## Rules

- Always read the full roadmap before entering plan mode.
- Do not expand scope beyond what the roadmap phase specifies.
- Follow all instructions in CLAUDE.md (TDD, worktree workflow, git conventions, AC verification).
- If the roadmap has no Status table (older roadmaps created before this convention), add one before proceeding.
