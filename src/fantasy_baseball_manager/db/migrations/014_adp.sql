CREATE TABLE IF NOT EXISTS adp (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id    INTEGER NOT NULL REFERENCES player(id),
    season       INTEGER NOT NULL,
    provider     TEXT NOT NULL,
    overall_pick REAL NOT NULL,
    rank         INTEGER NOT NULL,
    positions    TEXT NOT NULL,
    as_of        TEXT NOT NULL DEFAULT '',
    loaded_at    TEXT,
    UNIQUE(player_id, season, provider, positions, as_of)
);
