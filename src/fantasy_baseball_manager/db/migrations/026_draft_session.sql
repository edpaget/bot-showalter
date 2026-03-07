CREATE TABLE draft_session (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    league      TEXT    NOT NULL,
    season      INTEGER NOT NULL,
    teams       INTEGER NOT NULL,
    format      TEXT    NOT NULL,
    user_team   INTEGER NOT NULL,
    roster_slots TEXT   NOT NULL,  -- JSON
    budget      INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL DEFAULT 'in_progress',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE draft_session_pick (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES draft_session(id),
    pick_number INTEGER NOT NULL,
    team        INTEGER NOT NULL,
    player_id   INTEGER NOT NULL,
    player_name TEXT    NOT NULL,
    position    TEXT    NOT NULL,
    price       INTEGER,
    UNIQUE(session_id, pick_number)
);
