#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <branch-name> [base-ref]

Create a git worktree for parallel development with Claude Code.

Arguments:
  branch-name   Name for the new branch and worktree directory
  base-ref      Starting point (default: HEAD)

The worktree is created as a sibling directory named <repo>-<branch>.
Dependencies are installed via uv sync, large data directories are
symlinked, and .claude/settings.local.json is copied if present.

Example:
  $(basename "$0") feature-x
  $(basename "$0") feature-x origin/main
EOF
    exit 1
}

[[ $# -ge 1 ]] || usage

BRANCH="$1"
BASE="${2:-HEAD}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
REPO_NAME="$(basename "$REPO_ROOT")"
WORKTREE_DIR="$(dirname "$REPO_ROOT")/${REPO_NAME}-${BRANCH}"

if [[ -d "$WORKTREE_DIR" ]]; then
    echo "Error: directory already exists: $WORKTREE_DIR" >&2
    exit 1
fi

echo "Creating worktree at $WORKTREE_DIR (branch: $BRANCH from $BASE)..."
git worktree add -b "$BRANCH" "$WORKTREE_DIR" "$BASE"

# Install dependencies in an independent .venv
echo "Installing dependencies..."
(cd "$WORKTREE_DIR" && uv sync --quiet)

# Symlink large shared directories
for dir in data artifacts; do
    if [[ -d "$REPO_ROOT/$dir" ]]; then
        ln -s "$REPO_ROOT/$dir" "$WORKTREE_DIR/$dir"
        echo "Symlinked $dir/"
    fi
done

# Copy machine-specific Claude Code overrides
if [[ -f "$REPO_ROOT/.claude/settings.local.json" ]]; then
    mkdir -p "$WORKTREE_DIR/.claude"
    cp "$REPO_ROOT/.claude/settings.local.json" "$WORKTREE_DIR/.claude/"
    echo "Copied .claude/settings.local.json"
fi

echo ""
echo "Worktree ready: $WORKTREE_DIR"
echo "Branch: $BRANCH (from $BASE)"
echo ""
echo "To start Claude Code:"
echo "  claude --cwd $WORKTREE_DIR"
