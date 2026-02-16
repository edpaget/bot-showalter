from fantasy_baseball_manager.domain.league_environment import LeagueEnvironment
from fantasy_baseball_manager.domain.level_factor import LevelFactor
from fantasy_baseball_manager.domain.minor_league_batting_stats import (
    MinorLeagueBattingStats,
)
from fantasy_baseball_manager.models.mle.model import MLEModel
from fantasy_baseball_manager.models.mle.types import DEFAULT_AGE_BENCHMARKS
from fantasy_baseball_manager.models.protocols import ModelConfig


# --- Fake repos (constructor injection, no monkeypatching) ---


class FakeMinorLeagueBattingStatsRepo:
    def __init__(self, data: list[MinorLeagueBattingStats] | None = None) -> None:
        self._data = data or []

    def upsert(self, stats: MinorLeagueBattingStats) -> int:
        return 1

    def get_by_player(self, player_id: int) -> list[MinorLeagueBattingStats]:
        return [s for s in self._data if s.player_id == player_id]

    def get_by_player_season(self, player_id: int, season: int) -> list[MinorLeagueBattingStats]:
        return [s for s in self._data if s.player_id == player_id and s.season == season]

    def get_by_season_level(self, season: int, level: str) -> list[MinorLeagueBattingStats]:
        return [s for s in self._data if s.season == season and s.level == level]


class FakeLeagueEnvironmentRepo:
    def __init__(self, data: list[LeagueEnvironment] | None = None) -> None:
        self._data = data or []

    def upsert(self, env: LeagueEnvironment) -> int:
        return 1

    def get_by_league_season_level(self, league: str, season: int, level: str) -> LeagueEnvironment | None:
        for env in self._data:
            if env.league == league and env.season == season and env.level == level:
                return env
        return None

    def get_by_season_level(self, season: int, level: str) -> list[LeagueEnvironment]:
        return [e for e in self._data if e.season == season and e.level == level]

    def get_by_season(self, season: int) -> list[LeagueEnvironment]:
        return [e for e in self._data if e.season == season]


class FakeLevelFactorRepo:
    def __init__(self, data: list[LevelFactor] | None = None) -> None:
        self._data = data or []

    def upsert(self, factor: LevelFactor) -> int:
        return 1

    def get_by_level_season(self, level: str, season: int) -> LevelFactor | None:
        for f in self._data:
            if f.level == level and f.season == season:
                return f
        return None

    def get_by_season(self, season: int) -> list[LevelFactor]:
        return [f for f in self._data if f.season == season]


# --- Fixtures ---


def _mlb_env(season: int = 2025) -> LeagueEnvironment:
    return LeagueEnvironment(
        league="MLB",
        season=season,
        level="MLB",
        runs_per_game=4.6,
        avg=0.248,
        obp=0.315,
        slg=0.400,
        k_pct=0.230,
        bb_pct=0.083,
        hr_per_pa=0.035,
        babip=0.300,
    )


def _aa_env(season: int = 2025) -> LeagueEnvironment:
    return LeagueEnvironment(
        league="Eastern League",
        season=season,
        level="AA",
        runs_per_game=5.2,
        avg=0.260,
        obp=0.330,
        slg=0.410,
        k_pct=0.220,
        bb_pct=0.085,
        hr_per_pa=0.030,
        babip=0.310,
    )


def _aaa_env(season: int = 2025) -> LeagueEnvironment:
    return LeagueEnvironment(
        league="International League",
        season=season,
        level="AAA",
        runs_per_game=5.0,
        avg=0.260,
        obp=0.335,
        slg=0.420,
        k_pct=0.215,
        bb_pct=0.088,
        hr_per_pa=0.032,
        babip=0.305,
    )


def _aa_factor(season: int = 2025) -> LevelFactor:
    return LevelFactor(
        level="AA",
        season=season,
        factor=0.68,
        k_factor=1.12,
        bb_factor=0.88,
        iso_factor=0.72,
        babip_factor=0.90,
    )


def _aaa_factor(season: int = 2025) -> LevelFactor:
    return LevelFactor(
        level="AAA",
        season=season,
        factor=0.80,
        k_factor=1.05,
        bb_factor=0.94,
        iso_factor=0.85,
        babip_factor=0.95,
    )


def _milb_stats(
    *,
    player_id: int = 1,
    season: int = 2025,
    level: str = "AA",
    league: str = "Eastern League",
    pa: int = 500,
    ab: int = 450,
    h: int = 135,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 20,
    bb: int = 40,
    so: int = 100,
    age: float = 22.0,
    hbp: int = 5,
    sf: int = 5,
) -> MinorLeagueBattingStats:
    return MinorLeagueBattingStats(
        player_id=player_id,
        season=season,
        level=level,
        league=league,
        team="Test Team",
        g=120,
        pa=pa,
        ab=ab,
        h=h,
        doubles=doubles,
        triples=triples,
        hr=hr,
        r=70,
        rbi=65,
        bb=bb,
        so=so,
        sb=10,
        cs=3,
        avg=h / ab if ab else 0.0,
        obp=0.340,
        slg=0.520,
        age=age,
        hbp=hbp,
        sf=sf,
    )


def _build_model(
    milb_data: list[MinorLeagueBattingStats] | None = None,
    envs: list[LeagueEnvironment] | None = None,
    factors: list[LevelFactor] | None = None,
) -> MLEModel:
    return MLEModel(
        milb_repo=FakeMinorLeagueBattingStatsRepo(milb_data or []),
        league_env_repo=FakeLeagueEnvironmentRepo(envs or []),
        level_factor_repo=FakeLevelFactorRepo(factors or []),
    )


class TestMLEModelProperties:
    def test_name(self) -> None:
        model = _build_model()
        assert model.name == "mle"

    def test_supported_operations(self) -> None:
        model = _build_model()
        assert model.supported_operations == frozenset({"prepare", "predict"})


class TestMLEModelPrepare:
    def test_prepare_returns_row_count(self) -> None:
        stats = [
            _milb_stats(player_id=1, season=2025),
            _milb_stats(player_id=2, season=2025),
        ]
        model = _build_model(
            milb_data=stats,
            envs=[_aa_env(), _mlb_env()],
            factors=[_aa_factor()],
        )
        config = ModelConfig(seasons=[2025], model_params={"season": 2025})
        result = model.prepare(config)
        assert result.rows_processed == 2
        assert result.model_name == "mle"


class TestMLEModelPredict:
    def test_predict_end_to_end(self) -> None:
        stats = [_milb_stats(player_id=1, season=2025)]
        model = _build_model(
            milb_data=stats,
            envs=[_aa_env(), _mlb_env()],
            factors=[_aa_factor()],
        )
        config = ModelConfig(
            seasons=[2025],
            model_params={"season": 2026, "mle_seasons": [2025]},
        )
        result = model.predict(config)
        assert result.model_name == "mle"
        assert len(result.predictions) == 1

        pred = result.predictions[0]
        assert pred["player_id"] == 1
        assert pred["season"] == 2026
        assert pred["player_type"] == "batter"
        # Verify key stats are present
        assert "pa" in pred
        assert "avg" in pred
        assert "obp" in pred
        assert "slg" in pred
        assert "k_pct" in pred
        assert "bb_pct" in pred
        assert "iso" in pred
        assert "babip" in pred
        assert "hr" in pred

    def test_predict_multi_level_season(self) -> None:
        stats = [
            _milb_stats(
                player_id=1, season=2025, level="AA", league="Eastern League", pa=200, ab=180, h=54, bb=15, so=40
            ),
            _milb_stats(
                player_id=1, season=2025, level="AAA", league="International League", pa=300, ab=270, h=81, bb=25, so=60
            ),
        ]
        model = _build_model(
            milb_data=stats,
            envs=[_aa_env(), _aaa_env(), _mlb_env()],
            factors=[_aa_factor(), _aaa_factor()],
        )
        config = ModelConfig(
            seasons=[2025],
            model_params={"season": 2026, "mle_seasons": [2025]},
        )
        result = model.predict(config)
        assert len(result.predictions) == 1
        pred = result.predictions[0]
        assert pred["player_id"] == 1
        # Multi-level blended stats should exist
        assert pred["pa"] > 0

    def test_predict_multi_season_weighting(self) -> None:
        stats = [
            _milb_stats(player_id=1, season=2024, pa=500, ab=450, h=135, bb=40, so=100),
            _milb_stats(player_id=1, season=2025, pa=500, ab=450, h=135, bb=40, so=100),
        ]
        envs = [_aa_env(2024), _mlb_env(2024), _aa_env(2025), _mlb_env(2025)]
        factors = [_aa_factor(2024), _aa_factor(2025)]

        model = _build_model(milb_data=stats, envs=envs, factors=factors)
        config = ModelConfig(
            seasons=[2024, 2025],
            model_params={"season": 2026, "mle_seasons": [2024, 2025]},
        )
        result = model.predict(config)
        assert len(result.predictions) == 1

    def test_predict_regression_applied(self) -> None:
        # Compare low PA (heavy regression) vs high PA (light regression)
        low_pa_stats = [_milb_stats(player_id=1, season=2025, pa=200, ab=180, h=54, bb=15, so=40)]
        high_pa_stats = [_milb_stats(player_id=2, season=2025, pa=600, ab=540, h=162, bb=45, so=120)]

        model_low = _build_model(
            milb_data=low_pa_stats,
            envs=[_aa_env(), _mlb_env()],
            factors=[_aa_factor()],
        )
        model_high = _build_model(
            milb_data=high_pa_stats,
            envs=[_aa_env(), _mlb_env()],
            factors=[_aa_factor()],
        )
        config = ModelConfig(
            seasons=[2025],
            model_params={"season": 2026, "mle_seasons": [2025]},
        )
        result_low = model_low.predict(config)
        result_high = model_high.predict(config)

        # Both should produce predictions
        assert len(result_low.predictions) == 1
        assert len(result_high.predictions) == 1

        # Low PA player should be regressed more toward MLB avg (0.230 K%)
        mlb_k = 0.230
        low_k = result_low.predictions[0]["k_pct"]
        high_k = result_high.predictions[0]["k_pct"]
        assert abs(low_k - mlb_k) < abs(high_k - mlb_k)

    def test_predict_skips_player_below_min_pa(self) -> None:
        stats = [_milb_stats(player_id=1, season=2025, pa=50, ab=45, h=12, bb=4, so=10)]
        model = _build_model(
            milb_data=stats,
            envs=[_aa_env(), _mlb_env()],
            factors=[_aa_factor()],
        )
        config = ModelConfig(
            seasons=[2025],
            model_params={"season": 2026, "mle_seasons": [2025]},
        )
        result = model.predict(config)
        assert len(result.predictions) == 0

    def test_predict_skips_missing_league_env(self) -> None:
        stats = [_milb_stats(player_id=1, season=2025)]
        # No AA league environment provided — only MLB env
        model = _build_model(
            milb_data=stats,
            envs=[_mlb_env()],
            factors=[_aa_factor()],
        )
        config = ModelConfig(
            seasons=[2025],
            model_params={"season": 2026, "mle_seasons": [2025]},
        )
        result = model.predict(config)
        # Player should be skipped because no league env for AA
        assert len(result.predictions) == 0

    def test_predict_none_age_config_skips_age_adjustment(self) -> None:
        stats = [_milb_stats(player_id=1, season=2025, age=20.0)]
        model = _build_model(
            milb_data=stats,
            envs=[_aa_env(), _mlb_env()],
            factors=[_aa_factor()],
        )

        # Without age_config in model_params
        config_no_age = ModelConfig(
            seasons=[2025],
            model_params={"season": 2026, "mle_seasons": [2025]},
        )
        result_no_age = model.predict(config_no_age)

        # With age_config
        config_with_age = ModelConfig(
            seasons=[2025],
            model_params={
                "season": 2026,
                "mle_seasons": [2025],
                "age_config": {
                    "benchmarks": DEFAULT_AGE_BENCHMARKS,
                    "young_bonus_per_year": 0.025,
                    "old_penalty_per_year": 0.010,
                    "peak_age": 27.0,
                    "development_rate_per_year": 0.006,
                    "min_multiplier": 0.85,
                    "max_multiplier": 1.25,
                },
            },
        )
        result_with_age = model.predict(config_with_age)

        # Both produce results, but they should differ
        assert len(result_no_age.predictions) == 1
        assert len(result_with_age.predictions) == 1
        # A young player at AA (age 20, benchmark 21) gets favorable age adjustment
        # so the age-adjusted prediction should have higher OBP
        assert result_with_age.predictions[0]["obp"] > result_no_age.predictions[0]["obp"]

    def test_predict_includes_age_in_output(self) -> None:
        stats = [_milb_stats(player_id=1, season=2025, age=22.5)]
        model = _build_model(
            milb_data=stats,
            envs=[_aa_env(), _mlb_env()],
            factors=[_aa_factor()],
        )
        config = ModelConfig(
            seasons=[2025],
            model_params={"season": 2026, "mle_seasons": [2025]},
        )
        result = model.predict(config)
        pred = result.predictions[0]
        assert pred["age"] == 22.5
