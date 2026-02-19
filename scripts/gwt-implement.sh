#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] <roadmap-name> <phase-number>

Dispatch a headless Claude Code agent to implement a roadmap phase in a worktree.

Arguments:
  roadmap-name   Name of the roadmap (matches docs/plans/<name>.md)
  phase-number   Phase number to implement

Options:
  --model <model>       Model to use (passed to claude -p)
  --base <ref>          Base ref for worktree (default: HEAD)
  --max-budget <usd>    Max budget in USD (passed to claude --max-budget-usd)
  --dry-run             Print prompt and command without executing

Example:
  $(basename "$0") shared-logic-dedup 1
  $(basename "$0") --model sonnet shared-logic-dedup 2
  $(basename "$0") --dry-run shared-logic-dedup 1
EOF
    exit 1
}

MODEL=""
BASE="HEAD"
MAX_BUDGET=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --max-budget) MAX_BUDGET="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        -*) echo "Error: unknown option $1" >&2; usage ;;
        *) break ;;
    esac
done

[[ $# -ge 2 ]] || usage

ROADMAP="$1"
PHASE="$2"
BRANCH="roadmap/${ROADMAP}/phase-${PHASE}"
PLAN_DOC="docs/plans/${ROADMAP}.md"

# Validate plan doc exists
if [[ ! -f "$REPO_ROOT/$PLAN_DOC" ]]; then
    echo "Error: plan document not found: $PLAN_DOC" >&2
    exit 1
fi

# Validate claude CLI is available
if ! command -v claude &>/dev/null; then
    echo "Error: claude CLI not found on PATH" >&2
    exit 1
fi

# Derive worktree path (same convention as gwt.sh)
REPO_NAME="$(basename "$REPO_ROOT")"
DIR_SUFFIX="${BRANCH//\//-}"
WORKTREE_DIR="$(dirname "$REPO_ROOT")/${REPO_NAME}-${DIR_SUFFIX}"

# Build prompt
PROMPT=$(cat <<'PROMPT_EOF'
Read CLAUDE.md for project conventions and development commands.

Read docs/plans/ROADMAP_PLACEHOLDER.md and find the Phase PHASE_PLACEHOLDER section.

Implement exactly what the phase specifies — no scope expansion beyond the plan.

Follow TDD: write a failing test first, then the minimum code to make it pass, then refactor. Run `uv run pytest` after each major step to verify.

When all steps are complete, run the full quality gate before committing:
  uv run ruff format src tests
  uv run ruff check src tests
  uv run ty check src tests
  uv run pytest --cov

Commit with conventional commit style. Always combine git add and git commit in a single chained command (e.g., `git add file1 file2 && git commit -m "..."`). Do not push to remote.
PROMPT_EOF
)
PROMPT="${PROMPT//ROADMAP_PLACEHOLDER/$ROADMAP}"
PROMPT="${PROMPT//PHASE_PLACEHOLDER/$PHASE}"

# Build command
CMD=(claude -p --output-format text --allowedTools "Read,Write,Edit,Glob,Grep,Bash")
[[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")
[[ -n "$MAX_BUDGET" ]] && CMD+=(--max-budget-usd "$MAX_BUDGET")

if [[ "$DRY_RUN" == true ]]; then
    echo "=== Prompt ==="
    echo "$PROMPT"
    echo ""
    echo "=== Command ==="
    echo "cd $WORKTREE_DIR && echo \"\$PROMPT\" | ${CMD[*]}"
    exit 0
fi

# Create worktree
echo "Creating worktree..."
"$SCRIPT_DIR/gwt.sh" "$BRANCH" "$BASE"

# Ensure logs directory exists
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${ROADMAP}-phase-${PHASE}.log"

# Launch headless agent in background
echo "$PROMPT" | "${CMD[@]}" --cwd "$WORKTREE_DIR" > "$LOG_FILE" 2>&1 &
PID=$!

echo ""
echo "Headless agent dispatched."
echo "  Worktree: $WORKTREE_DIR"
echo "  PID:      $PID"
echo "  Log:      $LOG_FILE"
echo ""
echo "Follow progress:"
echo "  tail -f $LOG_FILE"
