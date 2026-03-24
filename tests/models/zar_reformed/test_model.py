import dataclasses

from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    EligibilityRules,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.protocols import ModelConfig
from fantasy_baseball_manager.models.zar.model import ZarModel
from fantasy_baseball_manager.models.zar_reformed.model import ZarReformedModel
from fantasy_baseball_manager.services.player_eligibility import PlayerEligibilityService
from tests.fakes.repos import (
    FakePlayerRepo,
    FakePositionAppearanceRepo,
    FakeProjectionRepo,
    FakeValuationRepo,
)


def _standard_league() -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=260,
        roster_batters=3,
        roster_pitchers=2,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="r", name="Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="sv+hld", name="SV+HLD", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        roster_util=1,
        positions={"C": 1, "OF": 1},
    )


def _build_projections() -> list[Projection]:
    return [
        Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.BATTER,
            stat_json={"pa": 600, "hr": 40.0, "r": 100.0},
        ),
        Projection(
            player_id=2,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.BATTER,
            stat_json={"pa": 550, "hr": 20.0, "r": 70.0},
        ),
        Projection(
            player_id=3,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.BATTER,
            stat_json={"pa": 570, "hr": 10.0, "r": 50.0},
        ),
        # Starter with high IP, no saves
        Projection(
            player_id=4,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.PITCHER,
            stat_json={"ip": 200, "w": 15.0, "sv": 0.0, "hld": 0.0},
        ),
        # Reliever with moderate IP (65), high saves
        Projection(
            player_id=5,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.PITCHER,
            stat_json={"ip": 65, "w": 4.0, "sv": 35.0, "hld": 5.0},
        ),
        # Reliever with low IP (40), some saves — should be excluded by min_ip=60
        Projection(
            player_id=6,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.PITCHER,
            stat_json={"ip": 40, "w": 2.0, "sv": 15.0, "hld": 10.0},
        ),
    ]


def _build_appearances() -> list[PositionAppearance]:
    return [
        PositionAppearance(player_id=1, season=2025, position="C", games=100),
        PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        PositionAppearance(player_id=3, season=2025, position="OF", games=140),
    ]


def _build_reformed_model(
    projections: list[Projection] | None = None,
) -> tuple[ZarReformedModel, FakeValuationRepo]:
    val_repo = FakeValuationRepo()
    pos_repo = FakePositionAppearanceRepo(_build_appearances())
    proj_repo = FakeProjectionRepo(projections if projections is not None else _build_projections())
    service = PlayerEligibilityService(pos_repo)
    model = ZarReformedModel(
        projection_repo=proj_repo,
        position_repo=pos_repo,
        player_repo=FakePlayerRepo(),
        valuation_repo=val_repo,
        eligibility_service=service,
    )
    return model, val_repo


def _build_zar_model(
    projections: list[Projection] | None = None,
) -> tuple[ZarModel, FakeValuationRepo]:
    val_repo = FakeValuationRepo()
    pos_repo = FakePositionAppearanceRepo(_build_appearances())
    proj_repo = FakeProjectionRepo(projections if projections is not None else _build_projections())
    service = PlayerEligibilityService(pos_repo)
    model = ZarModel(
        projection_repo=proj_repo,
        position_repo=pos_repo,
        player_repo=FakePlayerRepo(),
        valuation_repo=val_repo,
        eligibility_service=service,
    )
    return model, val_repo


def _standard_config() -> ModelConfig:
    return ModelConfig(
        seasons=[2025],
        model_params={
            "league": _standard_league(),
            "projection_system": "steamer",
        },
        version="1.0",
    )


class TestZarReformedSvHldRemoved:
    def test_sv_hld_has_zero_weight(self) -> None:
        """SV+HLD should not affect the composite z-score or dollar ranking."""
        model, val_repo = _build_reformed_model()
        model.predict(_standard_config())
        pitcher_vals = [v for v in val_repo.upserted if v.player_type == "pitcher"]
        # SV+HLD category z-scores should still be computed (present in category_scores)
        for v in pitcher_vals:
            assert "sv+hld" in v.category_scores
        # But the starter (player 4) should be valued higher than the reliever (player 5)
        # because SV+HLD is zeroed and player 4 has more wins
        starter = next(v for v in pitcher_vals if v.player_id == 4)
        reliever = next(v for v in pitcher_vals if v.player_id == 5)
        assert starter.value > reliever.value


class TestZarReformedMinIpRaised:
    def test_min_ip_raised_excludes_low_ip_pitchers(self) -> None:
        """Pitchers with IP < 60 should be excluded by the reformed model."""
        model, val_repo = _build_reformed_model()
        model.predict(_standard_config())
        pitcher_ids = {v.player_id for v in val_repo.upserted if v.player_type == "pitcher"}
        assert 4 in pitcher_ids  # IP=200, included
        assert 5 in pitcher_ids  # IP=65, included (>= 60)
        assert 6 not in pitcher_ids  # IP=40, excluded (< 60)

    def test_min_ip_preserves_higher_existing_threshold(self) -> None:
        """If league already has min_ip > 60, the reformed model should keep it."""
        league = dataclasses.replace(
            _standard_league(),
            eligibility=EligibilityRules(min_ip=100),
        )
        model, val_repo = _build_reformed_model()
        config = ModelConfig(
            seasons=[2025],
            model_params={"league": league, "projection_system": "steamer"},
            version="1.0",
        )
        model.predict(config)
        pitcher_ids = {v.player_id for v in val_repo.upserted if v.player_type == "pitcher"}
        assert 4 in pitcher_ids  # IP=200
        assert 5 not in pitcher_ids  # IP=65 < 100
        assert 6 not in pitcher_ids  # IP=40 < 100


class TestZarReformedBackwardCompat:
    def test_baseline_zar_unchanged(self) -> None:
        """Running baseline zar produces identical output regardless of reformed model existence."""
        model_a, val_repo_a = _build_zar_model()
        model_a.predict(_standard_config())

        model_b, val_repo_b = _build_zar_model()
        model_b.predict(_standard_config())

        values_a = {v.player_id: v.value for v in val_repo_a.upserted}
        values_b = {v.player_id: v.value for v in val_repo_b.upserted}
        assert values_a == values_b

    def test_reformed_vs_baseline_different(self) -> None:
        """Reformed and baseline produce different valuations when SV+HLD is present."""
        zar_model, zar_repo = _build_zar_model()
        zar_model.predict(_standard_config())

        reformed_model, reformed_repo = _build_reformed_model()
        reformed_model.predict(_standard_config())

        zar_values = {v.player_id: v.value for v in zar_repo.upserted}
        reformed_values = {v.player_id: v.value for v in reformed_repo.upserted}
        # They should differ because reformed zeroes SV+HLD and raises min_ip
        assert zar_values != reformed_values

    def test_reformed_system_name(self) -> None:
        """Reformed model should use 'zar-reformed' as the valuation system."""
        model, val_repo = _build_reformed_model()
        model.predict(_standard_config())
        for v in val_repo.upserted:
            assert v.system == "zar-reformed"
