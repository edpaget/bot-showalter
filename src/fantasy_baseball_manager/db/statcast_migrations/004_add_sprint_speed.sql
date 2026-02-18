CREATE TABLE IF NOT EXISTS sprint_speed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mlbam_id INTEGER NOT NULL,
    season INTEGER NOT NULL,
    sprint_speed REAL,
    hp_to_1b REAL,
    bolts INTEGER,
    competitive_runs INTEGER,
    loaded_at TEXT,
    UNIQUE(mlbam_id, season)
);
