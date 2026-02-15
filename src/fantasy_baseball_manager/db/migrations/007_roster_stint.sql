CREATE TABLE IF NOT EXISTS roster_stint (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id  INTEGER NOT NULL REFERENCES player(id),
    team_id    INTEGER NOT NULL REFERENCES team(id),
    season     INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    end_date   TEXT,
    loaded_at  TEXT,
    UNIQUE(player_id, team_id, start_date)
);
