CREATE TABLE IF NOT EXISTS feature_candidate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    expression TEXT NOT NULL,
    player_type TEXT NOT NULL,
    min_pa INTEGER,
    min_ip REAL,
    created_at TEXT NOT NULL
);
