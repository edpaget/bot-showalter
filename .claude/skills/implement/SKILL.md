---
name: implement
description: Implement a phase from an existing roadmap. Reads the roadmap, enters plan mode, and guides the full implement-and-merge cycle. Use when the user asks to "implement phase N of X" or "start working on X".
argument-hint: <roadmap> [phase N]
---

# Implement Skill

Implement a specific phase from an existing roadmap document.

## CRITICAL — Worktree Requirement

ALL implementation work MUST happen inside a worktree. After plan-mode approval, the VERY FIRST action is calling `EnterWorktree`. Writing ANY code, tests, or file edits before `EnterWorktree` is a bug.

Plan mode erases the skill context you are reading right now. The ONLY way to ensure the worktree step happens is to write it into the plan. Your plan MUST begin with:

> **Step 0 — Worktree setup:** Call `EnterWorktree` with `name: "<roadmap>"` (e.g., `name: "preseason-spine"`). Wait for confirmation before proceeding. ALL subsequent steps happen inside the worktree.

And your plan MUST end with:

> **Final step — Land the phase:** Update the roadmap Status table to `done (<date>)`, update `docs/plans/INDEX.md` progress, commit the doc changes, and merge back to main: `git push . HEAD:main && git checkout main`.

If already in a worktree from a previous phase in the same session, skip `EnterWorktree` and ensure you are on a fresh branch.

## Behavior

1. Parse `$ARGUMENTS` to identify the roadmap name (kebab-case, matching a file in `docs/plans/`) and the phase number. If the phase number is omitted, read the roadmap's Status table and pick the first phase that is `not started`.
2. Read `docs/plans/<roadmap>.md` and extract the target phase's steps and acceptance criteria.
3. Verify the phase is not already marked `done` or `in progress` in the Status table. If it is, inform the user and stop.
4. Check if the phase has any blockers (other roadmap phases listed as dependencies in the Ordering section). If blocked, inform the user and stop.
5. Update the roadmap's Status table to mark the phase `in progress`.
6. Enter plan mode — explore the codebase, design the implementation approach against the roadmap's steps and acceptance criteria, and present the plan for user approval. Write the `EnterWorktree` and "Land the phase" steps into the plan exactly as specified in the **Worktree Requirement** section above.

## Argument parsing

- `$ARGUMENTS` may look like: `draft-board-export phase 3`, `draft-board-export 3`, `tier-generator`, `tier-generator phase 1`
- The roadmap name is everything before an optional `phase` keyword or trailing number.
- If no phase number is given, auto-select the next `not started` phase.

## Rules

- Always read the full roadmap before entering plan mode.
- **Never write code outside a worktree.** If you are about to write code and have not called `EnterWorktree`, stop and call it first.
- Do not expand scope beyond what the roadmap phase specifies.
- Follow all instructions in CLAUDE.md (TDD, worktree workflow, git conventions, AC verification).
- If the roadmap has no Status table (older roadmaps created before this convention), add one before proceeding.
