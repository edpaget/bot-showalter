CREATE TABLE IF NOT EXISTS feature_checkpoint (
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    player_type TEXT NOT NULL,
    feature_columns TEXT NOT NULL,   -- JSON array
    params TEXT NOT NULL,            -- JSON object
    target_results TEXT NOT NULL,    -- JSON object (same format as experiment table)
    experiment_id INTEGER NOT NULL REFERENCES experiment(id),
    created_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE(name, model)
);
