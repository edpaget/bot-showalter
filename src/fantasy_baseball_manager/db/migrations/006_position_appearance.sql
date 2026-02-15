CREATE TABLE IF NOT EXISTS position_appearance (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES player(id),
    season    INTEGER NOT NULL,
    position  TEXT NOT NULL,
    games     INTEGER NOT NULL,
    loaded_at TEXT,
    UNIQUE(player_id, season, position)
);
