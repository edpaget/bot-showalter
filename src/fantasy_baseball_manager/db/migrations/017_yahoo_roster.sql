CREATE TABLE IF NOT EXISTS yahoo_roster_snapshot (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team_key    TEXT NOT NULL,
    league_key  TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    as_of       TEXT NOT NULL,
    UNIQUE(team_key, league_key, season, week, as_of)
);

CREATE TABLE IF NOT EXISTS yahoo_roster_entry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id     INTEGER NOT NULL REFERENCES yahoo_roster_snapshot(id),
    player_id       INTEGER,
    yahoo_player_key TEXT NOT NULL,
    player_name     TEXT NOT NULL,
    position        TEXT NOT NULL,
    roster_status   TEXT NOT NULL,
    acquisition_type TEXT NOT NULL
);
