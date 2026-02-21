CREATE TABLE IF NOT EXISTS yahoo_player_map (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    yahoo_player_key  TEXT NOT NULL UNIQUE,
    player_id         INTEGER NOT NULL REFERENCES player(id),
    player_type       TEXT NOT NULL,
    yahoo_name        TEXT NOT NULL,
    yahoo_team        TEXT NOT NULL,
    yahoo_positions   TEXT NOT NULL
);
