# K8s Deployment Roadmap

Deploy the fantasy baseball manager's read-only services (Discord bot, live draft board, and future web UI) to a Kubernetes cluster. The gold data (SQLite databases and model artifacts) is built by a K8s Job that fetches from external sources, runs the ingest/train/predict pipeline, and writes to a PersistentVolume. Reader deployments mount the same PV read-only.

This keeps the write-heavy, API-dependent pipeline off the local machine while the serving layer stays lightweight — no database server, no ORM migration, just SQLite files on a shared volume.

## Status

| Phase | Status |
|-------|--------|
| 1 — Read-only DB mode | not started |
| 2 — Gold build script | not started |
| 3 — Containerize | not started |
| 4 — K8s manifests | not started |

## Phase 1: Read-only DB mode

Add a connection mode that opens SQLite databases without attempting writes (no WAL pragma, no migration execution, no schema_version table creation).

### Context

`create_connection()` unconditionally sets `PRAGMA journal_mode=WAL` and runs pending migrations — both of which write to the database. When the serving deployments mount the PV read-only, these writes will fail. The reader services (agent, discord bot, draft board) never need to write, so a read-only connection path is safe and sufficient.

### Steps

1. Add a `read_only: bool = False` parameter to `create_connection()`. When `True`, open with `f"file:{path}?mode=ro"` and `uri=True`, skip the WAL pragma, and skip `_run_migrations()`.
2. Do the same for `create_statcast_connection()` — it delegates to `create_connection()`, so the parameter threads through.
3. Add a `read_only` parameter to `ConnectionPool.__init__()` and pass it through to each `create_connection()` call.
4. Add a `--read-only` flag (or `FBM_READ_ONLY` env var) to the `fbm` CLI callback that propagates through the composition roots (`build_chat_context`, `build_draft_board_context`, etc.).
5. Write tests: open a DB in read-only mode, confirm SELECTs work, confirm INSERTs raise, confirm no WAL file is created on disk.

### Acceptance criteria

- `create_connection(path, read_only=True)` opens the DB without writing any files (no `-wal`, no `-shm`).
- Read queries work normally through repos.
- Write attempts raise `sqlite3.OperationalError`.
- `fbm chat --read-only` and `fbm discord --read-only` start successfully against an existing DB.
- Existing default behavior (`read_only=False`) is unchanged — all current tests pass without modification.

## Phase 2: Gold build script

Create a single script that runs the full ingest-through-valuations pipeline, producing a self-contained `data/` directory with `fbm.db` and `statcast.db` ready for read-only serving.

### Context

Today the pipeline is a manual sequence of ~15 CLI commands with season ranges, model names, and ordering constraints. Automating this into a single entry point is a prerequisite for running it as a K8s Job. The script should be idempotent (safe to re-run) and configurable for the target season.

### Steps

1. Create `scripts/build_gold.sh`. Accept a `PREDICT_SEASON` env var (default: current year) and a `HISTORY_START` env var (default: 2020). The script runs:
   - Foundation: `fbm ingest players`, `fbm ingest bio`
   - Per-season (HISTORY_START through PREDICT_SEASON-1): `fbm ingest batting`, `pitching`, `statcast`, `sprint-speed`, `appearances`, `roster`, `il`
   - MiLB: `fbm ingest milb-batting` for recent seasons
   - Compute: `fbm compute league-env`
   - ADP: `fbm ingest adp-fetch --season $PREDICT_SEASON`
   - Third-party projections: `fbm import` for Steamer/ZiPS CSVs (if present in a configurable directory)
   - Consensus PT: `scripts/generate_consensus_pt.py`
   - Models: `fbm prepare`, `fbm train`, `fbm predict` for each model in the pipeline (marcel, statcast-gbm, statcast-gbm-preseason, composite-full, etc.)
   - Valuations: `fbm predict zar --season $PREDICT_SEASON`
2. Use `set -euo pipefail` so any step failure aborts the build.
3. Add a `--skip-statcast` flag (or `SKIP_STATCAST=1` env var) for faster builds that omit the ~1.8GB statcast ingest. The statcast-gbm models won't work, but marcel and basic valuations will.
4. Add basic logging: timestamp each step, print a summary at the end (elapsed time, DB file sizes).
5. Document the expected external dependencies: network access to FanGraphs, Baseball Savant, MLB Stats API, FantasyPros.

### Acceptance criteria

- `PREDICT_SEASON=2026 ./scripts/build_gold.sh` produces `data/fbm.db` and `data/statcast.db` from an empty `data/` directory.
- The resulting DB supports `fbm draft board --season 2026 --read-only` and `fbm chat --read-only`.
- `SKIP_STATCAST=1` completes faster and still produces a usable DB (marcel + zar valuations work).
- The script is idempotent — running it twice produces the same result.
- Third-party projection CSVs are optional — the script succeeds without them (skipping import steps).

## Phase 3: Containerize

Build a Docker image that can run both the gold build pipeline (write mode) and the reader services (read-only mode).

### Context

A single image simplifies the K8s manifests — the Job and Deployments use the same image with different entrypoints. The image needs `uv`, Python 3.14, the project dependencies, and the source code. Data lives on the PV, not in the image, keeping image size small (~500MB for deps vs ~2GB+ if data were baked in).

### Steps

1. Create `Dockerfile` at the project root:
   - Base: `python:3.14-slim`
   - Install `uv` from the official image (`COPY --from=ghcr.io/astral-sh/uv:latest`)
   - Copy `pyproject.toml`, `uv.lock`, `fbm.toml`, `src/`, `scripts/`
   - `uv sync --frozen --no-dev` (skip dev deps in production image)
   - Set `ENTRYPOINT ["uv", "run", "fbm"]`
2. Create `.dockerignore` excluding `data/`, `artifacts/`, `notebooks/`, `htmlcov/`, `.git/`, `__pycache__/`, `.venv/`.
3. Verify the image builds and can run `fbm --help`, `fbm list`, `fbm discord --help`.
4. Verify the gold build script works inside the container: mount an empty volume at `/app/data`, run `scripts/build_gold.sh`, confirm the DB is populated.
5. Verify read-only serving works: mount a pre-built data volume read-only, run `fbm discord --read-only`.
6. Add a `Makefile` or document the build/push commands: `docker build -t ghcr.io/<user>/fbm:latest .` and `docker push`.

### Acceptance criteria

- `docker build .` succeeds and produces an image under 600MB (excluding data).
- `docker run <image> --help` prints the fbm CLI help.
- `docker run -v ./data:/app/data <image> discord --read-only` starts the Discord bot (fails only on missing token, not on import errors or missing deps).
- The gold build script runs to completion inside the container with a mounted volume.
- `.dockerignore` prevents data files and dev artifacts from entering the image.

## Phase 4: K8s manifests

Create the Kubernetes resources to run the gold build as a Job/CronJob and deploy the reader services.

### Context

The target cluster is a personal K8s cluster. The manifests should be straightforward YAML — no Helm charts or Kustomize overlays needed at this scale. The core resources are: a PVC for the gold data, a Job (or CronJob) that populates it, and Deployments for the Discord bot and draft board server.

### Steps

1. Create `k8s/` directory with the following manifests:
   - `namespace.yaml` — `fbm` namespace.
   - `pvc.yaml` — PersistentVolumeClaim `fbm-data` (5Gi, ReadWriteOnce or ReadWriteMany depending on cluster storage class). Include a second PVC `fbm-artifacts` for model artifacts.
   - `secret.yaml` — template for `FBM_DISCORD_TOKEN` and `ANTHROPIC_API_KEY` (values redacted, instructions in comments).
   - `configmap.yaml` — `fbm.toml` contents (or mount from a ConfigMap).
   - `job-gold-build.yaml` — Job that runs `scripts/build_gold.sh` with the data PVC mounted read-write. Set resource requests/limits appropriate for the ingest workload (memory for statcast, CPU for model training). Include `restartPolicy: OnFailure` with a backoff limit.
   - `cronjob-gold-build.yaml` — CronJob wrapper around the Job, scheduled weekly (or manually triggered via `kubectl create job --from=cronjob/fbm-gold-build`).
   - `deployment-discord.yaml` — Deployment running `fbm discord --read-only --data-dir /data`. Mounts data PVC read-only. Pulls secrets for tokens. Single replica, no horizontal scaling needed.
   - `deployment-draft.yaml` — Deployment running `fbm draft live --read-only --host 0.0.0.0 --port 8000 --season 2026 --data-dir /data`. Mounts data PVC read-only.
   - `service-draft.yaml` — ClusterIP Service exposing port 8000.
   - `ingress-draft.yaml` — Ingress routing a hostname to the draft service (TLS via cert-manager if available).
2. Document the deployment sequence: create namespace, apply secrets, apply PVC, run the gold build Job, then deploy the reader services.
3. Add a `k8s/README.md` with setup instructions, prerequisites (container registry access, storage class), and common operations (trigger a rebuild, check logs, restart services).
4. Add health check endpoints to the Discord bot and draft server (or use TCP liveness probes) so K8s can detect and restart unhealthy pods.

### Acceptance criteria

- `kubectl apply -f k8s/` creates all resources without errors (given secrets are populated).
- The gold build Job runs to completion and populates the PVC.
- The Discord bot Deployment starts and connects to Discord (given a valid token).
- The draft board Deployment starts and serves HTTP on port 8000.
- The Ingress routes external traffic to the draft board.
- Pods restart automatically if they crash (liveness probes or default restart policy).
- `kubectl create job --from=cronjob/fbm-gold-build manual-build` triggers an on-demand rebuild.

## Ordering

Phases 1 → 2 → 3 → 4, strictly sequential. Phase 1 is a prerequisite for everything — without read-only mode, the reader services can't mount a read-only volume. Phase 2 depends on the existing CLI commands (no code dependency on phase 1, but the build script won't be tested end-to-end in a container until phase 3). Phase 3 is a prerequisite for phase 4 — K8s needs a container image.

Phase 1 is independently useful even without K8s — it makes it safer to open the DB from multiple processes locally (e.g., running the agent while the ingest is updating).

This roadmap has no hard dependencies on other roadmaps. The web-ui-foundation roadmap would benefit from this infrastructure (deploy FastAPI + React via the same K8s pattern), but neither blocks the other.
