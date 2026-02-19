# Worktree Workflow Roadmap

Streamline parallel Claude Code development by fixing settings propagation
and automating worktree creation.

## Phase 1: Split Claude Code settings (local vs project-scoped)

Move shared permissions and hooks from `.claude/settings.local.json` to
`.claude/settings.json` (git-tracked). This ensures every worktree and
every Claude Code session inherits the allow-list and hooks automatically.

### Steps

1. Create `.claude/settings.json` containing the `permissions.allow` list
   and the `hooks.PostToolUse` ruff-format hook from the current
   `settings.local.json`.
2. Reduce `settings.local.json` to an empty object (`{}`), keeping the
   file so git continues to ignore it via `.gitignore`.
3. Verify `.claude/settings.json` is not excluded by `.gitignore`.
4. Commit the new settings file.

### Acceptance criteria

- `claude` launched in a fresh worktree picks up permissions and hooks
  without any manual approval of the allow-listed commands.
- `settings.local.json` remains available for machine-specific overrides.

## Phase 2: Worktree creation script

Add a shell script (`scripts/gwt.sh`) that automates worktree setup so
spinning up a parallel roadmap is a single command.

### Steps

1. Create `scripts/gwt.sh` implementing:
   - `git worktree add -b <branch> <path> <base-ref>` with sensible
     defaults (sibling directory, HEAD as base).
   - `uv sync` in the new worktree to create an independent `.venv/`.
   - Symlink `data/` and `artifacts/` back to the main worktree to avoid
     duplicating large files.
   - Copy `.claude/settings.local.json` if it exists (for any
     machine-specific overrides).
   - Print a summary with the path and a ready-to-paste `claude` command.
2. Make the script executable and commit.

### Acceptance criteria

- Running `./scripts/gwt.sh feature-x` from the main checkout creates a
  working worktree at `../fbm-feature-x` with dependencies installed,
  data symlinked, and Claude Code settings available.

## Phase 3: Worktree cleanup helper

Add a companion `scripts/gwt-remove.sh` that tears down a worktree
cleanly.

### Steps

1. Create `scripts/gwt-remove.sh` implementing:
   - `git worktree remove <path>` (safe removal).
   - `git worktree prune` to clean stale metadata.
   - Optionally delete the branch if it has been merged
     (`--delete-branch` flag).
2. Commit the script.

### Acceptance criteria

- Running `./scripts/gwt-remove.sh ../fbm-feature-x` removes the
  worktree directory, prunes metadata, and optionally deletes the branch.

## Ordering

Phases are independent but listed in priority order. Phase 1 is the
highest-value change — it eliminates the permissions re-approval pain
with no workflow change. Phases 2-3 are convenience automation.
