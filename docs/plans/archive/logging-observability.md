# Logging & Observability Roadmap

**Created:** 2026-02-17
**Status:** Proposed
**Goal:** Add structured logging throughout the codebase to provide visibility into
external calls, data ingestion, model operations, and feature materialization.

## Motivation

The codebase currently has zero logging. All observability relies on CLI print
output, exception propagation, and the `LoadLog` audit table. This makes it
difficult to diagnose failures in external API calls, understand data ingestion
progress, or trace model training behavior. Adding structured logging will
improve debuggability and provide an operational foundation for future monitoring.

## Constraints

- Use the stdlib `logging` module — no third-party logging frameworks.
- Log levels should follow standard semantics: DEBUG for internals, INFO for
  operational milestones, WARNING for recoverable issues, ERROR for failures.
- Logging must not change any public interfaces or domain model signatures.
- Avoid logging sensitive data (API keys, full player datasets).
- Keep log messages concise and structured (include relevant IDs, counts, timing).

## Phase 1 — Logging Foundation

Set up the logging infrastructure and configure it from the CLI entry point.

- Add a `logging.basicConfig()` call in `cli/app.py` (or a dedicated
  `_logging.py` module) that configures format, level, and output.
- Accept a `--verbose` / `-v` flag on the top-level Typer app to toggle between
  INFO (default) and DEBUG.
- Suppress noisy third-party loggers (e.g., `httpx`, `pybaseball`) to WARNING
  unless `--verbose` is set.
- Verify: `fbm --verbose predict ...` shows DEBUG output; default shows INFO only.

## Phase 2 — Ingestion Logging

Add logging to the data ingestion layer where external I/O happens.

- `ingest/loader.py` (`StatsLoader.load`): Log INFO at start ("Loading {source_type}
  from {source_detail}"), INFO on completion ("Loaded {n} rows into {target_table}
  in {elapsed}s"), ERROR on failure.
- `ingest/pybaseball_source.py`: Log DEBUG before each pybaseball call with
  parameters, DEBUG on response size.
- `ingest/mlb_milb_stats_source.py` and `mlb_transactions_source.py`: Log DEBUG
  for HTTP requests (URL, params), INFO for response status and row count.
- `ingest/il_parser.py`: Log WARNING for unparseable rows with context.

## Phase 3 — Model Operation Logging

Add logging to model lifecycle operations.

- `models/run_manager.py`: Log INFO when a run starts (system, version, config
  summary) and completes (metrics summary, artifact path).
- Each model's `prepare()`, `train()`, `predict()`: Log INFO with season range,
  player count, and timing. Log DEBUG for per-stat metrics.
- `models/gbm_training.py`: Log INFO for training start/completion with
  hyperparameter summary. Log DEBUG for per-fold CV results.
- `cli/_dispatcher.py`: Log DEBUG when dispatching operation to model.

## Phase 4 — Feature & Database Logging

Add logging to feature materialization and database operations.

- `features/assembler.py` (`SqliteDatasetAssembler`): Log INFO when materializing
  a feature set (name, version hash, row count). Log DEBUG for individual feature
  SQL generation.
- `features/transforms/`: Log DEBUG for each transform computation with row counts.
- `db/connection.py`: Log INFO on database creation/migration. Log DEBUG for
  migration steps applied.
- `db/pool.py`: Log WARNING when pool exhaustion is approached. Log DEBUG for
  connection checkout/return.

## Phase 5 — Service Layer Logging

Add logging to business logic services.

- `services/projection_evaluator.py`: Log INFO with evaluation summary (system,
  season, stat count, headline metrics).
- `services/cohort.py`: Log DEBUG for cohort filtering (input/output counts).
- `services/valuation_lookup.py`, `projection_lookup.py`: Log DEBUG for lookups
  with query parameters and result count.

## Success Criteria

- Every external I/O operation (HTTP, pybaseball, file reads) has at least one
  INFO log line with timing.
- Every model lifecycle operation logs start/completion at INFO.
- Default output is clean (INFO only); `--verbose` reveals DEBUG detail.
- No test changes required — logging is additive and does not affect behavior.
- Existing CLI output formatting is unchanged.
