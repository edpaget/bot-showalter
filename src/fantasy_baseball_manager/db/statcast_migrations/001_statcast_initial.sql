CREATE TABLE IF NOT EXISTS statcast_pitch (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_pk INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    batter_id INTEGER NOT NULL,
    pitcher_id INTEGER NOT NULL,
    at_bat_number INTEGER NOT NULL,
    pitch_number INTEGER NOT NULL,
    pitch_type TEXT,
    release_speed REAL,
    release_spin_rate REAL,
    pfx_x REAL,
    pfx_z REAL,
    plate_x REAL,
    plate_z REAL,
    zone INTEGER,
    events TEXT,
    description TEXT,
    launch_speed REAL,
    launch_angle REAL,
    hit_distance_sc REAL,
    barrel INTEGER,
    estimated_ba_using_speedangle REAL,
    estimated_woba_using_speedangle REAL,
    loaded_at TEXT,
    UNIQUE(game_pk, at_bat_number, pitch_number)
);
CREATE INDEX IF NOT EXISTS idx_sc_pitcher_date ON statcast_pitch(pitcher_id, game_date);
CREATE INDEX IF NOT EXISTS idx_sc_batter_date ON statcast_pitch(batter_id, game_date);
CREATE INDEX IF NOT EXISTS idx_sc_game ON statcast_pitch(game_pk);
