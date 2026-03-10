import dataclasses

from fantasy_baseball_manager.domain import (
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    Trainable,
)
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    EligibilityRules,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.zar.model import ZarModel
from fantasy_baseball_manager.services.player_eligibility import PlayerEligibilityService
from tests.fakes.repos import (
    FakePitchingStatsRepo,
    FakePlayerRepo,
    FakePositionAppearanceRepo,
    FakeProjectionRepo,
    FakeValuationRepo,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
            CategoryConfig(
                key="avg",
                name="AVG",
                stat_type=StatType.RATE,
                direction=Direction.HIGHER,
                numerator="h",
                denominator="ab",
            ),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="sv", name="Saves", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
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
            player_type="batter",
            stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0, "avg": 0.291},
        ),
        Projection(
            player_id=2,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 550, "hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0, "avg": 0.280},
        ),
        Projection(
            player_id=3,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 570, "hr": 10.0, "r": 50.0, "h": 130.0, "ab": 520.0, "avg": 0.250},
        ),
        Projection(
            player_id=4,
            season=2025,
            system="steamer",
            version="v1",
            player_type="pitcher",
            stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
        ),
        Projection(
            player_id=5,
            season=2025,
            system="steamer",
            version="v1",
            player_type="pitcher",
            stat_json={"ip": 70, "w": 8.0, "sv": 30.0},
        ),
    ]


def _build_appearances() -> list[PositionAppearance]:
    return [
        PositionAppearance(player_id=1, season=2025, position="C", games=100),
        PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        PositionAppearance(player_id=3, season=2025, position="OF", games=140),
    ]


def _build_model(
    projections: list[Projection] | None = None,
    appearances: list[PositionAppearance] | None = None,
    valuation_repo: FakeValuationRepo | None = None,
) -> tuple[ZarModel, FakeValuationRepo]:
    val_repo = valuation_repo or FakeValuationRepo()
    pos_repo = FakePositionAppearanceRepo(_build_appearances() if appearances is None else appearances)
    return (
        ZarModel(
            projection_repo=FakeProjectionRepo(_build_projections() if projections is None else projections),
            player_repo=FakePlayerRepo(),
            position_repo=pos_repo,
            valuation_repo=val_repo,
            eligibility_service=PlayerEligibilityService(pos_repo),
        ),
        val_repo,
    )


def _standard_config() -> ModelConfig:
    return ModelConfig(
        seasons=[2025],
        model_params={
            "league": _standard_league(),
            "projection_system": "steamer",
        },
        version="1.0",
    )


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestZarModelProtocol:
    def test_is_model(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Model)

    def test_is_predictable(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Predictable)

    def test_is_not_trainable(self) -> None:
        model, _ = _build_model()
        assert not isinstance(model, Trainable)

    def test_is_not_evaluable(self) -> None:
        model, _ = _build_model()
        assert not isinstance(model, Evaluable)

    def test_is_not_finetuneable(self) -> None:
        model, _ = _build_model()
        assert not isinstance(model, FineTunable)

    def test_name(self) -> None:
        model, _ = _build_model()
        assert model.name == "zar"

    def test_supported_operations(self) -> None:
        model, _ = _build_model()
        assert model.supported_operations == frozenset({"predict"})

    def test_artifact_type(self) -> None:
        model, _ = _build_model()
        assert model.artifact_type == "none"


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------


class TestZarModelPredict:
    def test_predict_returns_predict_result(self) -> None:
        model, _ = _build_model()
        result = model.predict(_standard_config())
        assert result.model_name == "zar"

    def test_predict_produces_predictions_for_all_players(self) -> None:
        model, _ = _build_model()
        result = model.predict(_standard_config())
        player_ids = {p["player_id"] for p in result.predictions}
        assert player_ids == {1, 2, 3, 4, 5}

    def test_predict_persists_valuations(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        assert len(val_repo.upserted) == 5

    def test_predict_valuations_have_ranks(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        ranks = sorted(v.rank for v in val_repo.upserted)
        assert ranks == [1, 2, 3, 4, 5]

    def test_predict_valuations_ranked_by_value_descending(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        by_rank = sorted(val_repo.upserted, key=lambda v: v.rank)
        values = [v.value for v in by_rank]
        assert values == sorted(values, reverse=True)

    def test_predict_category_scores_present(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        for v in val_repo.upserted:
            assert isinstance(v.category_scores, dict)
            assert len(v.category_scores) > 0

    def test_predict_batter_categories_match_league(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        batter_vals = [v for v in val_repo.upserted if v.player_type == "batter"]
        for v in batter_vals:
            assert set(v.category_scores.keys()) == {"hr", "r", "avg"}

    def test_predict_pitcher_categories_match_league(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        pitcher_vals = [v for v in val_repo.upserted if v.player_type == "pitcher"]
        for v in pitcher_vals:
            assert set(v.category_scores.keys()) == {"w", "sv"}

    def test_predict_valuation_metadata(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        for v in val_repo.upserted:
            assert v.system == "zar"
            assert v.version == "1.0"
            assert v.projection_system == "steamer"
            assert v.season == 2025

    def test_predict_assigns_valid_positions(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        valid_positions = {"C", "OF", "UTIL"}
        batter_vals = [v for v in val_repo.upserted if v.player_type == "batter"]
        for v in batter_vals:
            assert v.position in valid_positions, f"Player {v.player_id} got position {v.position!r}"

    def test_predict_pitcher_position(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        pitcher_vals = [v for v in val_repo.upserted if v.player_type == "pitcher"]
        for v in pitcher_vals:
            assert v.position == "P"

    def test_predict_empty_projections(self) -> None:
        model, val_repo = _build_model(projections=[], appearances=[])
        result = model.predict(_standard_config())
        assert result.predictions == []
        assert len(val_repo.upserted) == 0

    def test_predict_batters_with_no_position_get_util(self) -> None:
        """Batters without position appearances should be eligible for util."""
        projections = [
            Projection(
                player_id=99,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 550, "hr": 25.0, "r": 80.0, "h": 150.0, "ab": 500.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        config = _standard_config()
        model.predict(config)
        assert len(val_repo.upserted) == 1
        assert val_repo.upserted[0].position == "UTIL"

    def test_predict_filters_by_projection_version(self) -> None:
        """When projection_version is set, only that version's projections are used."""
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v2",
                player_type="batter",
                stat_json={"pa": 580, "hr": 35.0, "r": 90.0, "h": 150.0, "ab": 530.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v2",
                player_type="pitcher",
                stat_json={"ip": 180, "w": 12.0, "sv": 5.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
                "projection_version": "v1",
            },
            version="1.0",
        )
        model.predict(config)
        player_ids = {v.player_id for v in val_repo.upserted}
        assert player_ids == {1, 4}
        assert all(v.projection_version == "v1" for v in val_repo.upserted)

    def test_predict_without_projection_version_uses_all(self) -> None:
        """Without projection_version, all versions for the system are loaded."""
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=2,
                season=2025,
                system="steamer",
                version="v2",
                player_type="batter",
                stat_json={"pa": 550, "hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        config = _standard_config()
        model.predict(config)
        player_ids = {v.player_id for v in val_repo.upserted}
        assert player_ids == {1, 2}

    def test_predict_excludes_zero_pa_batters(self) -> None:
        """Batters with pa=0 should be excluded from the valuation pool."""
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=2,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 0, "hr": 0.0, "r": 0.0, "h": 0.0, "ab": 0.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        model.predict(_standard_config())
        player_ids = {v.player_id for v in val_repo.upserted}
        assert 2 not in player_ids
        assert 1 in player_ids

    def test_predict_excludes_zero_ip_pitchers(self) -> None:
        """Pitchers with ip=0 should be excluded from the valuation pool."""
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
            ),
            Projection(
                player_id=5,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 0, "w": 0.0, "sv": 0.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        model.predict(_standard_config())
        player_ids = {v.player_id for v in val_repo.upserted}
        assert 5 not in player_ids
        assert 4 in player_ids

    def test_predict_no_position_batter_with_roster_util_zero(self) -> None:
        """With roster_util=0, batters without position data are penalized but still valued."""
        league = dataclasses.replace(_standard_league(), roster_util=0)
        # Players 1-3 from standard projections; player 99 has same stats as player 2 but no position
        projections = _build_projections()[:3] + [
            Projection(
                player_id=99,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 550, "hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0, "avg": 0.280},
            ),
        ]
        model, val_repo = _build_model(projections=projections)
        config = ModelConfig(
            seasons=[2025],
            model_params={"league": league, "projection_system": "steamer"},
            version="1.0",
        )
        model.predict(config)

        batter_vals = [v for v in val_repo.upserted if v.player_type == "batter"]
        player_ids = {v.player_id for v in batter_vals}
        assert 99 in player_ids  # Still gets a valuation

        # Player 2 has same stats but has OF position — should be valued higher
        positioned = next(v for v in batter_vals if v.player_id == 2)
        unpositioned = next(v for v in batter_vals if v.player_id == 99)
        assert positioned.value > unpositioned.value

    def test_predict_projection_version_filters_by_season(self) -> None:
        """projection_version + season filter: only matching season is used."""
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=2,
                season=2024,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 550, "hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
                "projection_version": "v1",
            },
            version="1.0",
        )
        model.predict(config)
        player_ids = {v.player_id for v in val_repo.upserted}
        assert player_ids == {1}


class TestZarModelPreloadedProjections:
    """Tests for passing pre-loaded projections via model_params."""

    def test_preloaded_projections_used_instead_of_repo(self) -> None:
        """When projections are in model_params, the model uses them directly."""
        # Build model with full projections in repo
        model, val_repo = _build_model()

        # Pass only a subset as pre-loaded projections
        subset = _build_projections()[:2]  # Only first 2 batters
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
                "projections": subset,
            },
            version="1.0",
        )
        result = model.predict(config)
        player_ids = {p["player_id"] for p in result.predictions}
        # Should only contain the 2 players from the subset
        assert player_ids == {1, 2}

    def test_empty_projections_list_falls_back_to_repo(self) -> None:
        """An empty projections list should fall back to reading from repo."""
        model, _ = _build_model()
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
                "projections": [],
            },
            version="1.0",
        )
        result = model.predict(config)
        player_ids = {p["player_id"] for p in result.predictions}
        assert player_ids == {1, 2, 3, 4, 5}

    def test_no_projections_key_reads_from_repo(self) -> None:
        """Without projections key, predictions use repo (unchanged behavior)."""
        model1, _ = _build_model()
        result1 = model1.predict(_standard_config())

        model2, _ = _build_model()
        config2 = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
            },
            version="1.0",
        )
        result2 = model2.predict(config2)

        values1 = {p["player_id"]: p["value"] for p in result1.predictions}
        values2 = {p["player_id"]: p["value"] for p in result2.predictions}
        assert values1 == values2


class TestZarModelEligibilityService:
    """Tests for PlayerEligibilityService integration into ZarModel."""

    def test_uses_injected_eligibility_service(self) -> None:
        """When an eligibility service is provided, ZarModel uses it for fallback."""
        # Position data only in 2024, projections for 2025
        appearances = [
            PositionAppearance(player_id=1, season=2024, position="C", games=100),
            PositionAppearance(player_id=2, season=2024, position="OF", games=150),
            PositionAppearance(player_id=3, season=2024, position="OF", games=140),
        ]
        pos_repo = FakePositionAppearanceRepo(appearances)
        service = PlayerEligibilityService(pos_repo)
        val_repo = FakeValuationRepo()
        model = ZarModel(
            projection_repo=FakeProjectionRepo(_build_projections()),
            position_repo=pos_repo,
            player_repo=FakePlayerRepo(),
            valuation_repo=val_repo,
            eligibility_service=service,
        )
        model.predict(_standard_config())
        batter_vals = [v for v in val_repo.upserted if v.player_type == "batter"]
        # With fallback to 2024 data, batters should have valid positions
        valid_positions = {"C", "OF", "UTIL"}
        for v in batter_vals:
            assert v.position in valid_positions

    def test_fallback_assigns_positions_for_future_season(self) -> None:
        """For 2026 projections with only 2025 position data, service falls back."""
        projections = [
            Projection(
                player_id=1,
                season=2026,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0, "avg": 0.291},
            ),
            Projection(
                player_id=2,
                season=2026,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 550, "hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0, "avg": 0.280},
            ),
            Projection(
                player_id=4,
                season=2026,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
            ),
        ]
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=100),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        pos_repo = FakePositionAppearanceRepo(appearances)
        service = PlayerEligibilityService(pos_repo)
        val_repo = FakeValuationRepo()
        model = ZarModel(
            projection_repo=FakeProjectionRepo(projections),
            position_repo=pos_repo,
            player_repo=FakePlayerRepo(),
            valuation_repo=val_repo,
            eligibility_service=service,
        )
        config = ModelConfig(
            seasons=[2026],
            model_params={"league": _standard_league(), "projection_system": "steamer"},
            version="1.0",
        )
        model.predict(config)
        batter_vals = [v for v in val_repo.upserted if v.player_type == "batter"]
        # Without fallback, all batters would be "util". With fallback to 2025,
        # at least one batter should have a non-util position (c or of).
        non_util = [v for v in batter_vals if v.position != "UTIL"]
        assert len(non_util) > 0, "Fallback should assign real positions, not all util"

    def test_eligibility_service_always_injected(self) -> None:
        """The helper always injects an eligibility service; verify it works."""
        model, val_repo = _build_model()
        model.predict(_standard_config())
        # Positions from same season via the injected service
        batter_vals = [v for v in val_repo.upserted if v.player_type == "batter"]
        assert len(batter_vals) == 3
        valid_positions = {"C", "OF", "UTIL"}
        for v in batter_vals:
            assert v.position in valid_positions


# ---------------------------------------------------------------------------
# Pitcher SP/RP split tests
# ---------------------------------------------------------------------------


def _sp_rp_league() -> LeagueSettings:
    """League with sp=2, rp=2, p=2 pitcher slots."""
    return dataclasses.replace(
        _standard_league(),
        roster_pitchers=6,
        pitcher_positions={"SP": 2, "RP": 2, "P": 2},
    )


class TestZarModelPitcherPositions:
    """Tests for SP/RP classification flowing through ZarModel.predict()."""

    def test_empty_pitcher_positions_matches_current_behavior(self) -> None:
        """Without pitcher_positions, all pitchers get position='p'."""
        model, val_repo = _build_model()
        model.predict(_standard_config())
        pitcher_vals = [v for v in val_repo.upserted if v.player_type == "pitcher"]
        for v in pitcher_vals:
            assert v.position == "P"

    def test_sp_rp_split_produces_separate_positions(self) -> None:
        """With pitcher_positions, pitchers get SP or RP based on stats."""
        pitching_stats = [
            PitchingStats(player_id=4, season=2025, source="fg", g=32, gs=32),  # SP
            PitchingStats(player_id=5, season=2025, source="fg", g=65, gs=0),  # RP
        ]
        pos_repo = FakePositionAppearanceRepo(_build_appearances())
        pitching_repo = FakePitchingStatsRepo(pitching_stats)
        service = PlayerEligibilityService(pos_repo, pitching_stats_repo=pitching_repo)
        val_repo = FakeValuationRepo()
        model = ZarModel(
            projection_repo=FakeProjectionRepo(_build_projections()),
            position_repo=pos_repo,
            player_repo=FakePlayerRepo(),
            valuation_repo=val_repo,
            eligibility_service=service,
        )
        league = _sp_rp_league()
        config = ModelConfig(
            seasons=[2025],
            model_params={"league": league, "projection_system": "steamer"},
            version="1.0",
        )
        model.predict(config)
        pitcher_vals = {v.player_id: v for v in val_repo.upserted if v.player_type == "pitcher"}
        # Player 4 (SP only) should be assigned sp or p
        assert pitcher_vals[4].position in {"SP", "P"}
        # Player 5 (RP only) should be assigned rp or p
        assert pitcher_vals[5].position in {"RP", "P"}

    def test_dual_eligible_pitcher_gets_best_position(self) -> None:
        """A dual SP/RP pitcher should be assigned whichever has lower replacement level."""
        pitching_stats = [
            PitchingStats(player_id=4, season=2025, source="fg", g=32, gs=20),  # dual
            PitchingStats(player_id=5, season=2025, source="fg", g=65, gs=0),  # RP
        ]
        pos_repo = FakePositionAppearanceRepo(_build_appearances())
        pitching_repo = FakePitchingStatsRepo(pitching_stats)
        service = PlayerEligibilityService(pos_repo, pitching_stats_repo=pitching_repo)
        val_repo = FakeValuationRepo()
        model = ZarModel(
            projection_repo=FakeProjectionRepo(_build_projections()),
            position_repo=pos_repo,
            player_repo=FakePlayerRepo(),
            valuation_repo=val_repo,
            eligibility_service=service,
        )
        league = _sp_rp_league()
        config = ModelConfig(
            seasons=[2025],
            model_params={"league": league, "projection_system": "steamer"},
            version="1.0",
        )
        model.predict(config)
        dual_val = next(v for v in val_repo.upserted if v.player_id == 4)
        # Dual-eligible pitcher should get a valid pitcher position
        assert dual_val.position in {"SP", "RP", "P"}


# ---------------------------------------------------------------------------
# min_pa / min_ip cutoff tests
# ---------------------------------------------------------------------------


class TestZarModelPlayingTimeCutoffs:
    def test_min_pa_filters_low_pa_batters(self) -> None:
        """Batters with PA below min_pa are excluded from valuations."""
        league = dataclasses.replace(
            _standard_league(),
            eligibility=EligibilityRules(min_pa=100),
        )
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=2,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 50, "hr": 2.0, "r": 5.0, "h": 10.0, "ab": 45.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        config = ModelConfig(
            seasons=[2025],
            model_params={"league": league, "projection_system": "steamer"},
            version="1.0",
        )
        model.predict(config)
        player_ids = {v.player_id for v in val_repo.upserted}
        assert 1 in player_ids
        assert 2 not in player_ids  # PA=50 < min_pa=100

    def test_min_ip_filters_low_ip_pitchers(self) -> None:
        """Pitchers with IP below min_ip are excluded from valuations."""
        league = dataclasses.replace(
            _standard_league(),
            eligibility=EligibilityRules(min_ip=50),
        )
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
            ),
            Projection(
                player_id=5,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 10, "w": 1.0, "sv": 2.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        config = ModelConfig(
            seasons=[2025],
            model_params={"league": league, "projection_system": "steamer"},
            version="1.0",
        )
        model.predict(config)
        player_ids = {v.player_id for v in val_repo.upserted}
        assert 4 in player_ids
        assert 5 not in player_ids  # IP=10 < min_ip=50

    def test_default_min_pa_zero_includes_all_nonzero(self) -> None:
        """Default (min_pa=0) still excludes PA=0 but includes PA=1."""
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 1, "hr": 0.0, "r": 0.0, "h": 0.0, "ab": 1.0},
            ),
            Projection(
                player_id=2,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 0, "hr": 0.0, "r": 0.0, "h": 0.0, "ab": 0.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        model.predict(_standard_config())
        player_ids = {v.player_id for v in val_repo.upserted}
        assert 1 in player_ids  # PA=1 included
        assert 2 not in player_ids  # PA=0 excluded


# ---------------------------------------------------------------------------
# Variance correction tests
# ---------------------------------------------------------------------------


class TestZarModelSystemName:
    def test_predict_default_system_is_zar(self) -> None:
        """Without valuation_system param, all valuations use system='zar'."""
        model, val_repo = _build_model()
        model.predict(_standard_config())
        for v in val_repo.upserted:
            assert v.system == "zar"

    def test_predict_custom_valuation_system(self) -> None:
        """With valuation_system in model_params, valuations use that system."""
        model, val_repo = _build_model()
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
                "valuation_system": "zar-injury-risk",
            },
            version="1.0",
        )
        model.predict(config)
        for v in val_repo.upserted:
            assert v.system == "zar-injury-risk"


class TestZarModelCategoryWeights:
    def test_category_weights_passed_through(self) -> None:
        """Weights from model_params reach the engine, producing different dollar values."""
        model_without, val_repo_without = _build_model()
        model_without.predict(_standard_config())

        model_with, val_repo_with = _build_model()
        config_with = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
                "category_weights": {"sv": 0.0},
            },
            version="1.0",
        )
        model_with.predict(config_with)

        values_without = {v.player_id: v.value for v in val_repo_without.upserted}
        values_with = {v.player_id: v.value for v in val_repo_with.upserted}
        assert values_without != values_with

    def test_no_category_weights_backward_compatible(self) -> None:
        """Without category_weights, behavior is identical to baseline."""
        model_a, val_repo_a = _build_model()
        model_a.predict(_standard_config())

        model_b, val_repo_b = _build_model()
        model_b.predict(_standard_config())

        values_a = {v.player_id: v.value for v in val_repo_a.upserted}
        values_b = {v.player_id: v.value for v in val_repo_b.upserted}
        assert values_a == values_b


class TestZarModelVarianceCorrection:
    def test_predict_with_stdev_overrides(self) -> None:
        """When _stdev_overrides is in model_params, dollar values differ from without."""
        model_without, val_repo_without = _build_model()
        model_without.predict(_standard_config())

        model_with, val_repo_with = _build_model()
        # Pre-computed stdev overrides (as the CLI would produce)
        config_with = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
                "_stdev_overrides": {"hr": 50.0, "r": 50.0, "avg": 50.0, "w": 50.0, "sv": 50.0},
            },
            version="1.0",
        )
        model_with.predict(config_with)

        values_without = {v.player_id: v.value for v in val_repo_without.upserted}
        values_with = {v.player_id: v.value for v in val_repo_with.upserted}
        assert values_without != values_with

    def test_predict_without_stdev_overrides_unchanged(self) -> None:
        """Without _stdev_overrides, behavior is identical to baseline."""
        model_a, val_repo_a = _build_model()
        model_a.predict(_standard_config())

        model_b, val_repo_b = _build_model()
        model_b.predict(_standard_config())

        values_a = {v.player_id: v.value for v in val_repo_a.upserted}
        values_b = {v.player_id: v.value for v in val_repo_b.upserted}
        assert values_a == values_b
