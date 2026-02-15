-- Add operation column to model_run (train, predict, etc.)
-- SQLite cannot drop an inline UNIQUE constraint, so recreate the table.

CREATE TABLE model_run_new (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    system                TEXT NOT NULL,
    version               TEXT NOT NULL,
    operation             TEXT NOT NULL DEFAULT 'train',
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
    UNIQUE(system, version, operation)
);

INSERT INTO model_run_new
    (id, system, version, operation, train_dataset_id, validation_dataset_id,
     holdout_dataset_id, config_json, metrics_json, artifact_type,
     artifact_path, git_commit, tags_json, created_at)
SELECT
    id, system, version, 'train', train_dataset_id, validation_dataset_id,
    holdout_dataset_id, config_json, metrics_json, artifact_type,
    artifact_path, git_commit, tags_json, created_at
FROM model_run;

DROP TABLE model_run;

ALTER TABLE model_run_new RENAME TO model_run;
