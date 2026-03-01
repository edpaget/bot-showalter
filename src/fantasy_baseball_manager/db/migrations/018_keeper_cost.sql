CREATE TABLE IF NOT EXISTS keeper_cost (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id       INTEGER NOT NULL REFERENCES player(id),
    season          INTEGER NOT NULL,
    league          TEXT NOT NULL,
    cost            REAL NOT NULL,
    years_remaining INTEGER NOT NULL DEFAULT 1,
    source          TEXT NOT NULL,
    loaded_at       TEXT,
    UNIQUE(player_id, season, league)
);
