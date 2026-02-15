CREATE TABLE IF NOT EXISTS il_stint (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id        INTEGER NOT NULL REFERENCES player(id),
    season           INTEGER NOT NULL,
    start_date       TEXT NOT NULL,
    il_type          TEXT NOT NULL,
    end_date         TEXT,
    days             INTEGER,
    injury_location  TEXT,
    transaction_type TEXT,
    loaded_at        TEXT,
    UNIQUE(player_id, start_date, il_type)
);
