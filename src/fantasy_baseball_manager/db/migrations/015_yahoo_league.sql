CREATE TABLE IF NOT EXISTS yahoo_league (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    league_key  TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    season      INTEGER NOT NULL,
    num_teams   INTEGER NOT NULL,
    draft_type  TEXT NOT NULL,
    is_keeper   INTEGER NOT NULL DEFAULT 0,
    game_key    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS yahoo_team (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    team_key          TEXT NOT NULL UNIQUE,
    league_key        TEXT NOT NULL REFERENCES yahoo_league(league_key),
    team_id           INTEGER NOT NULL,
    name              TEXT NOT NULL,
    manager_name      TEXT NOT NULL,
    is_owned_by_user  INTEGER NOT NULL DEFAULT 0
);
