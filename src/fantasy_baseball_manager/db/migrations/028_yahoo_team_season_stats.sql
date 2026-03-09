CREATE TABLE IF NOT EXISTS yahoo_team_season_stats (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    team_key         TEXT    NOT NULL,
    league_key       TEXT    NOT NULL,
    season           INTEGER NOT NULL,
    team_name        TEXT    NOT NULL,
    final_rank       INTEGER NOT NULL,
    stat_values_json TEXT    NOT NULL,
    UNIQUE(team_key, league_key, season)
);
