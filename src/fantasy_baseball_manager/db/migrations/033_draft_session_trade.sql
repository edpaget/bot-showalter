CREATE TABLE IF NOT EXISTS draft_session_trade (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES draft_session(id),
    trade_number  INTEGER NOT NULL,
    team_a        INTEGER NOT NULL,
    team_b        INTEGER NOT NULL,
    team_a_gives  TEXT    NOT NULL,
    team_b_gives  TEXT    NOT NULL,
    UNIQUE(session_id, trade_number)
);
