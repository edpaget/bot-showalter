CREATE TABLE IF NOT EXISTS league_environment (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    league        TEXT NOT NULL,
    season        INTEGER NOT NULL,
    level         TEXT NOT NULL,
    runs_per_game REAL NOT NULL,
    avg           REAL NOT NULL,
    obp           REAL NOT NULL,
    slg           REAL NOT NULL,
    k_pct         REAL NOT NULL,
    bb_pct        REAL NOT NULL,
    hr_per_pa     REAL NOT NULL,
    babip         REAL NOT NULL,
    loaded_at     TEXT,
    UNIQUE(league, season, level)
);

CREATE TABLE IF NOT EXISTS level_factor (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    level        TEXT NOT NULL,
    season       INTEGER NOT NULL,
    factor       REAL NOT NULL,
    k_factor     REAL NOT NULL,
    bb_factor    REAL NOT NULL,
    iso_factor   REAL NOT NULL,
    babip_factor REAL NOT NULL,
    loaded_at    TEXT,
    UNIQUE(level, season)
);

-- Seed level factors from Szymborski/Davenport research
-- factor: overall quality vs MLB. k/bb/iso/babip: component-specific translation factors.
-- k_factor > 1 means K% increases at MLB. Others < 1 mean rates decrease.
INSERT OR IGNORE INTO level_factor (level, season, factor, k_factor, bb_factor, iso_factor, babip_factor) VALUES
    ('AAA', 2024, 0.80, 1.15, 0.92, 0.85, 0.95),
    ('AA',  2024, 0.67, 1.22, 0.88, 0.78, 0.93),
    ('A+',  2024, 0.57, 1.28, 0.85, 0.72, 0.91),
    ('A',   2024, 0.42, 1.35, 0.80, 0.65, 0.88),
    ('ROK', 2024, 0.32, 1.42, 0.75, 0.58, 0.85);

-- Copy to 2023 and 2022 (same baseline, can be refined later)
INSERT OR IGNORE INTO level_factor (level, season, factor, k_factor, bb_factor, iso_factor, babip_factor)
    SELECT level, 2023, factor, k_factor, bb_factor, iso_factor, babip_factor FROM level_factor WHERE season = 2024;
INSERT OR IGNORE INTO level_factor (level, season, factor, k_factor, bb_factor, iso_factor, babip_factor)
    SELECT level, 2022, factor, k_factor, bb_factor, iso_factor, babip_factor FROM level_factor WHERE season = 2024;
