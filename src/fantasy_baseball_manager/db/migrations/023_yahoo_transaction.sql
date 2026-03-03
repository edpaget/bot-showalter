CREATE TABLE IF NOT EXISTS yahoo_transaction (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_key    TEXT NOT NULL UNIQUE,
    league_key         TEXT NOT NULL,
    type               TEXT NOT NULL,
    timestamp          TEXT NOT NULL,
    status             TEXT NOT NULL,
    trader_team_key    TEXT NOT NULL,
    tradee_team_key    TEXT
);

CREATE TABLE IF NOT EXISTS yahoo_transaction_player (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_key    TEXT NOT NULL REFERENCES yahoo_transaction(transaction_key),
    player_id          INTEGER,
    yahoo_player_key   TEXT NOT NULL,
    player_name        TEXT NOT NULL,
    source_team_key    TEXT,
    dest_team_key      TEXT,
    type               TEXT NOT NULL
);
