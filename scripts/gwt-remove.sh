#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [--delete-branch] <worktree-path>

Remove a git worktree created by gwt.sh.

Arguments:
  worktree-path    Path to the worktree directory

Options:
  --delete-branch  Also delete the branch if it has been merged

Uses --force because worktrees always contain generated files (.venv/,
symlinks, local settings) that git considers untracked.

Example:
  $(basename "$0") ../fbm-feature-x
  $(basename "$0") --delete-branch ../fbm-feature-x
EOF
    exit 1
}

DELETE_BRANCH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delete-branch) DELETE_BRANCH=true; shift ;;
        -h|--help) usage ;;
        -*) echo "Error: unknown option $1" >&2; usage ;;
        *) break ;;
    esac
done

[[ $# -ge 1 ]] || usage

WORKTREE_PATH="$1"

if [[ ! -d "$WORKTREE_PATH" ]]; then
    echo "Error: directory does not exist: $WORKTREE_PATH" >&2
    exit 1
fi

# Resolve the branch name before removing the worktree
BRANCH="$(git -C "$WORKTREE_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"

echo "Removing worktree at $WORKTREE_PATH..."
git worktree remove --force "$WORKTREE_PATH"

echo "Pruning stale worktree metadata..."
git worktree prune

if [[ "$DELETE_BRANCH" == true && -n "$BRANCH" && "$BRANCH" != "HEAD" ]]; then
    echo "Deleting branch $BRANCH..."
    git branch -d "$BRANCH"
fi

echo "Done."
