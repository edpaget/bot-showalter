CREATE INDEX IF NOT EXISTS idx_valuation_season_system_version ON valuation(season, system, version);
CREATE INDEX IF NOT EXISTS idx_adp_season_provider ON adp(season, provider);
