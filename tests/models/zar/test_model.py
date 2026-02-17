from typing import Any

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.models.protocols import (
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    Trainable,
)
from fantasy_baseball_manager.models.zar.model import ZarModel


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProjectionRepo:
    def __init__(self, projections: list[Projection]) -> None:
        self._projections = projections

    def upsert(self, projection: Projection) -> int:
        return 1

    def get_by_player_season(
        self, player_id: int, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        return [p for p in self._projections if p.player_id == player_id and p.season == season]

    def get_by_season(
        self, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        result = [p for p in self._projections if p.season == season]
        if system is not None:
            result = [p for p in result if p.system == system]
        return result

    def get_by_system_version(self, system: str, version: str) -> list[Projection]:
        return [p for p in self._projections if p.system == system and p.version == version]

    def upsert_distributions(self, projection_id: int, distributions: list[Any]) -> None:
        pass

    def get_distributions(self, projection_id: int) -> list[Any]:
        return []


class FakePlayerRepo:
    def __init__(self, players: list[Player] | None = None) -> None:
        self._players = players or []

    def upsert(self, player: Player) -> int:
        return 1

    def get_by_id(self, player_id: int) -> Player | None:
        return next((p for p in self._players if p.id == player_id), None)

    def get_by_mlbam_id(self, mlbam_id: int) -> Player | None:
        return None

    def get_by_bbref_id(self, bbref_id: str) -> Player | None:
        return None

    def search_by_name(self, name: str) -> list[Player]:
        return []

    def get_by_last_name(self, last_name: str) -> list[Player]:
        return []

    def all(self) -> list[Player]:
        return list(self._players)


class FakePositionAppearanceRepo:
    def __init__(self, appearances: list[PositionAppearance]) -> None:
        self._appearances = appearances

    def upsert(self, appearance: PositionAppearance) -> int:
        return 1

    def get_by_player(self, player_id: int) -> list[PositionAppearance]:
        return [a for a in self._appearances if a.player_id == player_id]

    def get_by_player_season(self, player_id: int, season: int) -> list[PositionAppearance]:
        return [a for a in self._appearances if a.player_id == player_id and a.season == season]

    def get_by_season(self, season: int) -> list[PositionAppearance]:
        return [a for a in self._appearances if a.season == season]


class FakeValuationRepo:
    def __init__(self) -> None:
        self.upserted: list[Valuation] = []

    def upsert(self, valuation: Valuation) -> int:
        self.upserted.append(valuation)
        return len(self.upserted)

    def get_by_player_season(self, player_id: int, season: int, system: str | None = None) -> list[Valuation]:
        result = [v for v in self.upserted if v.player_id == player_id and v.season == season]
        if system is not None:
            result = [v for v in result if v.system == system]
        return result

    def get_by_season(self, season: int, system: str | None = None) -> list[Valuation]:
        result = [v for v in self.upserted if v.season == season]
        if system is not None:
            result = [v for v in result if v.system == system]
        return result


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
        positions={"c": 1, "of": 1},
    )


def _build_projections() -> list[Projection]:
    return [
        Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0, "avg": 0.291},
        ),
        Projection(
            player_id=2,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0, "avg": 0.280},
        ),
        Projection(
            player_id=3,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"hr": 10.0, "r": 50.0, "h": 130.0, "ab": 520.0, "avg": 0.250},
        ),
        Projection(
            player_id=4,
            season=2025,
            system="steamer",
            version="v1",
            player_type="pitcher",
            stat_json={"w": 15.0, "sv": 0.0},
        ),
        Projection(
            player_id=5,
            season=2025,
            system="steamer",
            version="v1",
            player_type="pitcher",
            stat_json={"w": 8.0, "sv": 30.0},
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
    return (
        ZarModel(
            projection_repo=FakeProjectionRepo(_build_projections() if projections is None else projections),
            player_repo=FakePlayerRepo(),
            position_repo=FakePositionAppearanceRepo(_build_appearances() if appearances is None else appearances),
            valuation_repo=val_repo,
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
        valid_positions = {"c", "of", "util"}
        batter_vals = [v for v in val_repo.upserted if v.player_type == "batter"]
        for v in batter_vals:
            assert v.position in valid_positions, f"Player {v.player_id} got position {v.position!r}"

    def test_predict_pitcher_position(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        pitcher_vals = [v for v in val_repo.upserted if v.player_type == "pitcher"]
        for v in pitcher_vals:
            assert v.position == "p"

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
                stat_json={"hr": 25.0, "r": 80.0, "h": 150.0, "ab": 500.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        config = _standard_config()
        model.predict(config)
        assert len(val_repo.upserted) == 1
        assert val_repo.upserted[0].position == "util"

    def test_predict_filters_by_projection_version(self) -> None:
        """When projection_version is set, only that version's projections are used."""
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v2",
                player_type="batter",
                stat_json={"hr": 35.0, "r": 90.0, "h": 150.0, "ab": 530.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"w": 15.0, "sv": 0.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v2",
                player_type="pitcher",
                stat_json={"w": 12.0, "sv": 5.0},
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
                stat_json={"hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=2,
                season=2025,
                system="steamer",
                version="v2",
                player_type="batter",
                stat_json={"hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        config = _standard_config()
        model.predict(config)
        player_ids = {v.player_id for v in val_repo.upserted}
        assert player_ids == {1, 2}

    def test_predict_projection_version_filters_by_season(self) -> None:
        """projection_version + season filter: only matching season is used."""
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=2,
                season=2024,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0},
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
