CREATE TABLE IF NOT EXISTS player_alias (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alias_name  TEXT NOT NULL,
    player_id   INTEGER NOT NULL REFERENCES player(id),
    player_type TEXT NOT NULL DEFAULT '',
    source      TEXT,
    active_from INTEGER,
    active_to   INTEGER,
    UNIQUE(alias_name, player_id, player_type)
);

CREATE INDEX IF NOT EXISTS idx_player_alias_name ON player_alias(alias_name);
