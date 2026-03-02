CREATE TABLE IF NOT EXISTS experiment (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    hypothesis      TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    player_type     TEXT    NOT NULL,
    feature_diff    TEXT    NOT NULL,
    seasons         TEXT    NOT NULL,
    params          TEXT    NOT NULL,
    target_results  TEXT    NOT NULL,
    conclusion      TEXT    NOT NULL,
    tags            TEXT    NOT NULL DEFAULT '[]',
    parent_id       INTEGER REFERENCES experiment(id),
    FOREIGN KEY (parent_id) REFERENCES experiment(id)
);

CREATE INDEX IF NOT EXISTS idx_experiment_model ON experiment(model);
CREATE INDEX IF NOT EXISTS idx_experiment_timestamp ON experiment(timestamp);
CREATE INDEX IF NOT EXISTS idx_experiment_player_type ON experiment(player_type);
