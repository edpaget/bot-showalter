CREATE TABLE IF NOT EXISTS valuation (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id           INTEGER NOT NULL REFERENCES player(id),
    season              INTEGER NOT NULL,
    system              TEXT NOT NULL,
    version             TEXT NOT NULL,
    projection_system   TEXT NOT NULL,
    projection_version  TEXT NOT NULL,
    player_type         TEXT NOT NULL,
    position            TEXT NOT NULL,
    value               REAL NOT NULL,
    rank                INTEGER NOT NULL,
    category_scores_json TEXT NOT NULL,
    loaded_at           TEXT,
    UNIQUE(player_id, season, system, version, player_type)
);
