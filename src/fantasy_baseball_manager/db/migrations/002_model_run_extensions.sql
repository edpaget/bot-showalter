-- Recreate model_run with full provenance columns.
-- No production data exists yet, so DROP + CREATE is safe.
DROP TABLE IF EXISTS model_run;

CREATE TABLE model_run (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    system                TEXT NOT NULL,
    version               TEXT NOT NULL,
    train_dataset_id      INTEGER REFERENCES dataset(id),
    validation_dataset_id INTEGER REFERENCES dataset(id),
    holdout_dataset_id    INTEGER REFERENCES dataset(id),
    config_json           TEXT NOT NULL,
    metrics_json          TEXT,
    artifact_type         TEXT NOT NULL,
    artifact_path         TEXT,
    git_commit            TEXT,
    tags_json             TEXT,
    created_at            TEXT NOT NULL,
    UNIQUE(system, version)
);

-- Distinguish first-party vs third-party projections.
ALTER TABLE projection ADD COLUMN source_type TEXT NOT NULL DEFAULT 'first_party';
