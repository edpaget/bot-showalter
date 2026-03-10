CREATE TABLE IF NOT EXISTS league_keeper (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER NOT NULL REFERENCES player(id),
    season      INTEGER NOT NULL,
    league      TEXT NOT NULL,
    team_name   TEXT NOT NULL,
    cost        REAL,
    source      TEXT,
    UNIQUE(player_id, season, league)
);
