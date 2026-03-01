CREATE TABLE IF NOT EXISTS yahoo_draft_pick (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    league_key        TEXT NOT NULL,
    season            INTEGER NOT NULL,
    round             INTEGER NOT NULL,
    pick              INTEGER NOT NULL,
    team_key          TEXT NOT NULL,
    yahoo_player_key  TEXT NOT NULL,
    player_id         INTEGER,
    player_name       TEXT NOT NULL,
    position          TEXT NOT NULL,
    cost              INTEGER,
    UNIQUE(league_key, season, round, pick)
);
