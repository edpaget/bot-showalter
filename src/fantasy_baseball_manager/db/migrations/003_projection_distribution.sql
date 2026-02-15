CREATE TABLE IF NOT EXISTS projection_distribution (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    projection_id INTEGER NOT NULL REFERENCES projection(id),
    stat          TEXT NOT NULL,
    p10           REAL NOT NULL,
    p25           REAL NOT NULL,
    p50           REAL NOT NULL,
    p75           REAL NOT NULL,
    p90           REAL NOT NULL,
    mean          REAL,
    std           REAL,
    family        TEXT,
    UNIQUE(projection_id, stat)
);
