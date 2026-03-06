# Model Training Inspector Roadmap

Inspecting model training parameters currently requires ad-hoc SQLite queries against the `model_run` table. The existing `fbm runs show` command displays config/metrics as raw JSON and requires an exact `system/version` identifier. This roadmap adds a dedicated `fbm runs inspect` command that surfaces training parameters in a structured, readable format with convenient defaults (latest run, section filtering) and a `fbm runs diff` command for comparing configs across versions.

## Status

| Phase | Status |
|-------|--------|
| 1 --- Structured inspect command | done (2026-03-05) |
| 2 --- Config diff across versions | done (2026-03-05) |
| 3 --- Update fbm skill documentation | done (2026-03-05) |

## Phase 1: Structured inspect command

Add `fbm runs inspect` that displays training configuration and metrics in a well-organized, section-based format with a "latest" shortcut.

### Context

`fbm runs show` already retrieves and prints a `ModelRunRecord`, but it dumps `config_json` and `metrics_json` as raw indented JSON in a single Rich table. To answer "what seasons/hyperparameters did I train statcast-gbm with?", you must already know the exact version string and then scan through unformatted JSON. The new command addresses three gaps:

1. **Latest-run shortcut** --- `fbm runs inspect statcast-gbm` (no version) should show the most recent training run.
2. **Sectioned output** --- Break `config_json` into logical sections (training seasons, hyperparameters, data paths) and `metrics_json` into a metrics section, each with labeled headers.
3. **Section filtering** --- `--section config` or `--section metrics` to show only one part.

### Steps

1. Add a `get_latest(system, operation)` method to `SqliteModelRunRepo` that returns the most recent `ModelRunRecord` for a system/operation pair (ORDER BY created_at DESC LIMIT 1). Write tests against an in-memory SQLite database.
2. Add a `print_run_inspect()` output function in `cli/_output/_runs.py` that renders a `ModelRunRecord` with:
   - A header panel showing system, version, operation, created_at, git_commit.
   - A "Config" section that prints `config_json` keys grouped into subsections: "Seasons" (`seasons`), "Model Params" (`model_params`), and "Paths" (`data_dir`, `artifacts_dir`, remaining keys).
   - A "Metrics" section that prints `metrics_json` as a two-column key/value table with rounded floats.
   - A "Tags" section if tags exist.
   - Accept an optional `section` filter parameter to limit output to one section.
   Write unit tests that verify the output contains expected section headers and values.
3. Add the `inspect` command to `runs_app` in `cli/commands/runs.py`:
   - Positional `system` argument (e.g., `statcast-gbm`).
   - Optional `--version` flag; when omitted, use `get_latest()`.
   - Optional `--section` flag (`config`, `metrics`, `tags`, or omit for all).
   - `--operation` flag defaulting to `train`.
### Acceptance criteria

- `fbm runs inspect statcast-gbm` shows the latest training run with sectioned output.
- `fbm runs inspect statcast-gbm --version v1.0.0` shows a specific version.
- `fbm runs inspect statcast-gbm --section metrics` shows only the metrics section.
- `get_latest()` returns `None` when no runs exist for a system and is covered by tests.
- `print_run_inspect()` output includes section headers ("Config", "Metrics") and is covered by tests.

## Phase 2: Config diff across versions

Add `fbm runs diff` to compare training configs and metrics between two runs of the same (or different) systems.

### Context

When iterating on model training --- changing seasons, tuning hyperparameters, adding features --- it is useful to see exactly what changed between two runs. Currently this requires two `runs show` calls and manual comparison.

### Steps

1. Add a `diff_records(a: ModelRunRecord, b: ModelRunRecord)` function in `cli/_output/_runs.py` that computes a structured diff of `config_json` and `metrics_json`:
   - For each top-level key, show "added", "removed", or "changed (old -> new)".
   - For nested dicts (e.g., `model_params`), recurse one level.
   - For `metrics_json`, show the delta (e.g., `rmse: 0.45 -> 0.42 (-0.03)`).
   Write unit tests covering added/removed/changed keys and nested diffs.
2. Add a `print_run_diff()` function that renders the diff output as a Rich table with color-coded changes (green for improved metrics, red for worse, yellow for changed config).
3. Add the `diff` command to `runs_app`:
   - Two positional arguments: `run_a` and `run_b`, each in `system/version` format (or just `system` to use latest).
   - `--operation` flag defaulting to `train`.
   - Resolve each argument using `get_latest()` or `get()` as appropriate.
4. Write integration-style tests that create two runs with different configs and verify the diff output.

### Acceptance criteria

- `fbm runs diff statcast-gbm/v1 statcast-gbm/v2` shows config and metric differences.
- `fbm runs diff statcast-gbm statcast-gbm/v1` compares latest against a specific version.
- Changed, added, and removed keys are clearly labeled in the output.
- Metric deltas show direction and magnitude.
- All diff logic is covered by unit tests.

## Phase 3: Update fbm skill documentation

Add the new `runs inspect` and `runs diff` commands to the fbm SKILL.md so Claude knows how to use them.

### Context

The fbm skill file (`.claude/skills/fbm/SKILL.md`) documents every CLI command so Claude can map natural-language requests to the correct `uv run fbm` invocation. New commands won't be discoverable unless added to both the "Available commands" section and the "Argument mapping" examples.

### Steps

1. Add a "Model run inspection" subsection under "Available commands" in `.claude/skills/fbm/SKILL.md` documenting:
   - `uv run fbm runs inspect <system> [--version <version>] [--section config|metrics|tags] [--operation <op>]`
   - `uv run fbm runs diff <run_a> <run_b> [--operation <op>]` (where each run is `system/version` or just `system` for latest)
   - Brief description and examples for each.
2. Add argument mapping entries for natural-language patterns like:
   - "what parameters was statcast-gbm trained with" -> `uv run fbm runs inspect statcast-gbm`
   - "show training config for statcast-gbm v2" -> `uv run fbm runs inspect statcast-gbm --version v2`
   - "show metrics for the latest marcel training run" -> `uv run fbm runs inspect marcel --section metrics`
   - "what changed between statcast-gbm v1 and v2" -> `uv run fbm runs diff statcast-gbm/v1 statcast-gbm/v2`
   - "diff latest statcast-gbm against v1" -> `uv run fbm runs diff statcast-gbm statcast-gbm/v1`
3. Update the skill's `description` field in the YAML frontmatter to mention inspecting training parameters (add "inspect model training parameters" to the list).

### Acceptance criteria

- `.claude/skills/fbm/SKILL.md` documents both `runs inspect` and `runs diff` with usage, examples, and argument mappings.
- The YAML `description` field mentions training parameter inspection.
- Existing command documentation is unchanged.

## Ordering

Phase 1 is the core deliverable and has no dependencies. Phase 2 depends on `get_latest()` from phase 1. Phase 3 depends on phases 1 and 2 being complete (so the documented commands actually exist). All phases are small; they can be implemented in a single session if desired.
