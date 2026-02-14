CREATE TABLE IF NOT EXISTS player (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name_first TEXT NOT NULL,
    name_last TEXT NOT NULL,
    mlbam_id INTEGER UNIQUE,
    fangraphs_id INTEGER UNIQUE,
    bbref_id TEXT UNIQUE,
    retro_id TEXT UNIQUE,
    bats TEXT,
    throws TEXT,
    birth_date TEXT,
    position TEXT
);

CREATE TABLE IF NOT EXISTS team (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    abbreviation TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    league TEXT NOT NULL,
    division TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batting_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES player(id),
    season INTEGER NOT NULL,
    team_id INTEGER REFERENCES team(id),
    source TEXT NOT NULL,
    pa INTEGER,
    ab INTEGER,
    h INTEGER,
    doubles INTEGER,
    triples INTEGER,
    hr INTEGER,
    rbi INTEGER,
    r INTEGER,
    sb INTEGER,
    cs INTEGER,
    bb INTEGER,
    so INTEGER,
    hbp INTEGER,
    sf INTEGER,
    sh INTEGER,
    gdp INTEGER,
    ibb INTEGER,
    avg REAL,
    obp REAL,
    slg REAL,
    ops REAL,
    woba REAL,
    wrc_plus REAL,
    war REAL,
    loaded_at TEXT,
    UNIQUE(player_id, season, source)
);

CREATE TABLE IF NOT EXISTS pitching_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES player(id),
    season INTEGER NOT NULL,
    team_id INTEGER REFERENCES team(id),
    source TEXT NOT NULL,
    w INTEGER,
    l INTEGER,
    era REAL,
    g INTEGER,
    gs INTEGER,
    sv INTEGER,
    hld INTEGER,
    ip REAL,
    h INTEGER,
    er INTEGER,
    hr INTEGER,
    bb INTEGER,
    so INTEGER,
    whip REAL,
    k_per_9 REAL,
    bb_per_9 REAL,
    fip REAL,
    xfip REAL,
    war REAL,
    loaded_at TEXT,
    UNIQUE(player_id, season, source)
);

CREATE TABLE IF NOT EXISTS projection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES player(id),
    season INTEGER NOT NULL,
    system TEXT NOT NULL,
    version TEXT NOT NULL,
    player_type TEXT NOT NULL,
    pa INTEGER, ab INTEGER, h INTEGER, doubles INTEGER, triples INTEGER,
    hr INTEGER, rbi INTEGER, r INTEGER, sb INTEGER, cs INTEGER,
    bb INTEGER, so INTEGER, hbp INTEGER, sf INTEGER, sh INTEGER,
    gdp INTEGER, ibb INTEGER, avg REAL, obp REAL, slg REAL,
    ops REAL, woba REAL, wrc_plus REAL, war REAL,
    w INTEGER, l INTEGER, era REAL, g INTEGER, gs INTEGER,
    sv INTEGER, hld INTEGER, ip REAL, er INTEGER,
    whip REAL, k_per_9 REAL, bb_per_9 REAL, fip REAL, xfip REAL,
    loaded_at TEXT,
    source_type TEXT NOT NULL DEFAULT 'first_party',
    UNIQUE(player_id, season, system, version, player_type)
);

CREATE TABLE IF NOT EXISTS feature_set (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT,
    source_query TEXT,
    created_at TEXT,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_set_id INTEGER NOT NULL REFERENCES feature_set(id),
    name TEXT NOT NULL,
    split TEXT,
    table_name TEXT,
    row_count INTEGER,
    seasons TEXT,
    created_at TEXT,
    params_json TEXT
);

CREATE TABLE IF NOT EXISTS model_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system TEXT NOT NULL,
    version TEXT NOT NULL,
    train_dataset_id INTEGER NOT NULL REFERENCES dataset(id),
    validation_dataset_id INTEGER REFERENCES dataset(id),
    holdout_dataset_id INTEGER REFERENCES dataset(id),
    metrics_json TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS load_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_detail TEXT NOT NULL,
    target_table TEXT NOT NULL,
    rows_loaded INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT
);
