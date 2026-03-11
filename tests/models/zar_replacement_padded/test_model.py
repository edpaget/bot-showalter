from fantasy_baseball_manager.domain import (
    Model,
    ModelConfig,
    Predictable,
)
from fantasy_baseball_manager.domain.il_stint import ILStint
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models import get
from fantasy_baseball_manager.models.zar.model import ZarModel
from fantasy_baseball_manager.models.zar_injury_risk.model import ZarInjuryRiskModel
from fantasy_baseball_manager.models.zar_replacement_padded.model import ZarReplacementPaddedModel
from fantasy_baseball_manager.services.injury_profiler import InjuryProfiler
from fantasy_baseball_manager.services.player_eligibility import PlayerEligibilityService
from fantasy_baseball_manager.services.replacement_padding_service import ReplacementPaddingService
from tests.fakes.repos import (
    FakePlayerRepo,
    FakePositionAppearanceRepo,
    FakeProjectionRepo,
    FakeValuationRepo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeILStintRepo:
    def __init__(self, stints: list[ILStint] | None = None) -> None:
        self._stints = stints or []

    def upsert(self, stint: ILStint) -> int:
        return 1

    def get_by_player(self, player_id: int) -> list[ILStint]:
        return [s for s in self._stints if s.player_id == player_id]

    def get_by_player_season(self, player_id: int, season: int) -> list[ILStint]:
        return [s for s in self._stints if s.player_id == player_id and s.season == season]

    def get_by_season(self, season: int) -> list[ILStint]:
        return [s for s in self._stints if s.season == season]

    def count_by_season(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for s in self._stints:
            counts[s.season] = counts.get(s.season, 0) + 1
        return counts


def _league() -> LeagueSettings:
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


def _projections() -> list[Projection]:
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


def _appearances() -> list[PositionAppearance]:
    return [
        PositionAppearance(player_id=1, season=2025, position="C", games=100),
        PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        PositionAppearance(player_id=3, season=2025, position="OF", games=140),
    ]


def _il_stints_for_player_1() -> list[ILStint]:
    """Player 1 has significant IL history across 2023-2024."""
    return [
        ILStint(player_id=1, season=2023, start_date="2023-05-01", il_type="10-day", days=30),
        ILStint(player_id=1, season=2024, start_date="2024-06-01", il_type="10-day", days=45),
    ]


def _build_model(
    stints: list[ILStint] | None = None,
    projections: list[Projection] | None = None,
) -> tuple[ZarReplacementPaddedModel, FakeValuationRepo]:
    proj_repo = FakeProjectionRepo(_projections() if projections is None else projections)
    pos_repo = FakePositionAppearanceRepo(_appearances())
    val_repo = FakeValuationRepo()
    player_repo = FakePlayerRepo()
    eligibility = PlayerEligibilityService(pos_repo)
    il_repo = FakeILStintRepo(stints)
    profiler = InjuryProfiler(player_repo=player_repo, il_stint_repo=il_repo)
    model = ZarReplacementPaddedModel(
        projection_repo=proj_repo,
        position_repo=pos_repo,
        player_repo=player_repo,
        valuation_repo=val_repo,
        eligibility_service=eligibility,
        injury_profiler=profiler,
        replacement_padder=ReplacementPaddingService(),
    )
    return model, val_repo


def _build_injury_risk_model(
    stints: list[ILStint] | None = None,
) -> tuple[ZarInjuryRiskModel, FakeValuationRepo]:
    proj_repo = FakeProjectionRepo(_projections())
    pos_repo = FakePositionAppearanceRepo(_appearances())
    val_repo = FakeValuationRepo()
    player_repo = FakePlayerRepo()
    eligibility = PlayerEligibilityService(pos_repo)
    il_repo = FakeILStintRepo(stints)
    profiler = InjuryProfiler(player_repo=player_repo, il_stint_repo=il_repo)
    model = ZarInjuryRiskModel(
        projection_repo=proj_repo,
        position_repo=pos_repo,
        player_repo=player_repo,
        valuation_repo=val_repo,
        eligibility_service=eligibility,
        injury_profiler=profiler,
    )
    return model, val_repo


def _build_plain_zar() -> tuple[ZarModel, FakeValuationRepo]:
    proj_repo = FakeProjectionRepo(_projections())
    pos_repo = FakePositionAppearanceRepo(_appearances())
    val_repo = FakeValuationRepo()
    model = ZarModel(
        projection_repo=proj_repo,
        position_repo=pos_repo,
        player_repo=FakePlayerRepo(),
        valuation_repo=val_repo,
        eligibility_service=PlayerEligibilityService(pos_repo),
    )
    return model, val_repo


def _config() -> ModelConfig:
    return ModelConfig(
        seasons=[2025],
        model_params={
            "league": _league(),
            "projection_system": "steamer",
            "use_optimal_assignment": False,
        },
        version="1.0",
    )


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestZarReplacementPaddedProtocol:
    def test_is_model(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Model)

    def test_is_predictable(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Predictable)

    def test_name(self) -> None:
        model, _ = _build_model()
        assert model.name == "zar-replacement-padded"

    def test_supported_operations(self) -> None:
        model, _ = _build_model()
        assert model.supported_operations == frozenset({"predict"})


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------


class TestZarReplacementPaddedPredict:
    def test_predict_produces_valuations_with_correct_system(self) -> None:
        """All persisted valuations should have system='zar-replacement-padded'."""
        model, val_repo = _build_model(stints=_il_stints_for_player_1())
        model.predict(_config())
        assert len(val_repo.upserted) > 0
        for v in val_repo.upserted:
            assert v.system == "zar-replacement-padded"

    def test_predict_delegates_to_zar(self) -> None:
        """The model should produce ranked valuations (full pipeline)."""
        model, _ = _build_model(stints=_il_stints_for_player_1())
        result = model.predict(_config())
        assert result.model_name == "zar-replacement-padded"
        assert len(result.predictions) == 5
        ranks = sorted(p["rank"] for p in result.predictions)
        assert ranks == [1, 2, 3, 4, 5]

    def test_elite_injury_prone_hitter_gains_value_vs_injury_risk(self) -> None:
        """Player 1 (heavy IL history) should have higher value under replacement-padded
        than under simple injury-risk discount, because replacement padding fills
        missed time with replacement-level production instead of zeroing it out."""
        stints = _il_stints_for_player_1()
        config = _config()

        padded_model, padded_repo = _build_model(stints=stints)
        padded_model.predict(config)
        padded_values = {v.player_id: v.value for v in padded_repo.upserted}

        ir_model, ir_repo = _build_injury_risk_model(stints=stints)
        ir_model.predict(config)
        ir_values = {v.player_id: v.value for v in ir_repo.upserted}

        assert padded_values[1] > ir_values[1]

    def test_no_injury_history_matches_plain_zar(self) -> None:
        """With no IL stints, results should match plain ZAR values."""
        plain_model, plain_val_repo = _build_plain_zar()
        plain_model.predict(_config())
        plain_values = {v.player_id: v.value for v in plain_val_repo.upserted}

        model, val_repo = _build_model(stints=[])
        model.predict(_config())
        padded_values = {v.player_id: v.value for v in val_repo.upserted}

        assert plain_values == padded_values

    def test_healthy_players_still_valued(self) -> None:
        """When some players have injury history, all healthy players are
        still included in the output with valid valuations."""
        stints = _il_stints_for_player_1()
        config = _config()

        padded_model, padded_repo = _build_model(stints=stints)
        padded_model.predict(config)
        padded_values = {v.player_id: v.value for v in padded_repo.upserted}

        # All healthy players should be present
        for pid in [2, 3, 4, 5]:
            assert pid in padded_values, f"Healthy player {pid} should be valued"

    def test_registered_and_discoverable(self) -> None:
        """Model should be discoverable via models.get()."""
        cls = get("zar-replacement-padded")
        assert cls is ZarReplacementPaddedModel
