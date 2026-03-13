-- Add player_type column and update unique constraint to include it.
-- SQLite doesn't support ALTER TABLE DROP CONSTRAINT, so we rebuild the table.
-- We store '' (empty string) instead of NULL so the UNIQUE constraint works
-- (SQLite treats NULLs as distinct in unique constraints).

CREATE TABLE league_keeper_new (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER NOT NULL REFERENCES player(id),
    season      INTEGER NOT NULL,
    league      TEXT NOT NULL,
    team_name   TEXT NOT NULL,
    cost        REAL,
    source      TEXT,
    player_type TEXT NOT NULL DEFAULT '',
    UNIQUE(player_id, season, league, player_type)
);

INSERT INTO league_keeper_new (id, player_id, season, league, team_name, cost, source, player_type)
SELECT id, player_id, season, league, team_name, cost, source, ''
FROM league_keeper;

DROP TABLE league_keeper;
ALTER TABLE league_keeper_new RENAME TO league_keeper;
