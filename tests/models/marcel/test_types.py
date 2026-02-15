from fantasy_baseball_manager.models.marcel.types import (
    LeagueAverages,
    MarcelConfig,
    MarcelProjection,
    SeasonLine,
)


class TestSeasonLine:
    def test_defaults(self) -> None:
        line = SeasonLine(stats={"hr": 30.0})
        assert line.stats == {"hr": 30.0}
        assert line.pa == 0
        assert line.ip == 0.0
        assert line.g == 0
        assert line.gs == 0

    def test_batter(self) -> None:
        line = SeasonLine(stats={"hr": 30.0, "h": 150.0}, pa=600)
        assert line.pa == 600
        assert line.ip == 0.0

    def test_pitcher(self) -> None:
        line = SeasonLine(stats={"so": 200.0, "er": 60.0}, ip=180.0, g=30, gs=30)
        assert line.ip == 180.0
        assert line.g == 30
        assert line.gs == 30

    def test_frozen(self) -> None:
        line = SeasonLine(stats={"hr": 30.0})
        try:
            line.pa = 100  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass


class TestLeagueAverages:
    def test_construction(self) -> None:
        avg = LeagueAverages(rates={"hr": 0.03, "h": 0.25})
        assert avg.rates["hr"] == 0.03

    def test_frozen(self) -> None:
        avg = LeagueAverages(rates={})
        try:
            avg.rates = {"hr": 0.1}  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass


class TestMarcelProjection:
    def test_defaults(self) -> None:
        proj = MarcelProjection(
            player_id=1,
            projected_season=2024,
            age=32,
            stats={"hr": 25.0},
            rates={"hr": 0.04},
        )
        assert proj.pa == 0
        assert proj.ip == 0.0

    def test_batter_projection(self) -> None:
        proj = MarcelProjection(
            player_id=1,
            projected_season=2024,
            age=32,
            stats={"hr": 25.0},
            rates={"hr": 0.04},
            pa=550,
        )
        assert proj.pa == 550
        assert proj.player_id == 1

    def test_pitcher_projection(self) -> None:
        proj = MarcelProjection(
            player_id=2,
            projected_season=2024,
            age=28,
            stats={"so": 180.0},
            rates={"so": 1.0},
            ip=180.0,
        )
        assert proj.ip == 180.0

    def test_frozen(self) -> None:
        proj = MarcelProjection(player_id=1, projected_season=2024, age=30, stats={}, rates={})
        try:
            proj.player_id = 2  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass


class TestMarcelConfig:
    def test_defaults(self) -> None:
        cfg = MarcelConfig()
        assert cfg.batting_weights == (5.0, 4.0, 3.0)
        assert cfg.pitching_weights == (3.0, 2.0, 1.0)
        assert cfg.batting_regression_pa == 1200.0
        assert cfg.pitching_regression_ip == 134.0
        assert cfg.batting_baseline_pa == 200.0
        assert cfg.pitching_starter_baseline_ip == 60.0
        assert cfg.pitching_reliever_baseline_ip == 25.0
        assert cfg.pa_weights == (0.5, 0.1)
        assert cfg.ip_weights == (0.5, 0.1)
        assert cfg.age_peak == 29
        assert cfg.age_improvement_rate == 0.006
        assert cfg.age_decline_rate == 0.003
        assert cfg.reliever_gs_ratio == 0.5
        assert len(cfg.batting_categories) == 12
        assert len(cfg.pitching_categories) == 8

    def test_custom_weights(self) -> None:
        cfg = MarcelConfig(batting_weights=(6.0, 3.0))
        assert cfg.batting_weights == (6.0, 3.0)

    def test_frozen(self) -> None:
        cfg = MarcelConfig()
        try:
            cfg.age_peak = 30  # type: ignore[misc]
            assert False, "Should raise"
        except AttributeError:
            pass
