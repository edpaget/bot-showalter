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
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.playing_time.engine import ResidualBuckets, ResidualPercentiles
from fantasy_baseball_manager.models.registry import get
from fantasy_baseball_manager.models.zar.model import ZarModel
from fantasy_baseball_manager.models.zar_distributional.model import ZarDistributionalModel
from fantasy_baseball_manager.services.distributional_valuation import run_distributional_zar
from fantasy_baseball_manager.services.injury_profiler import InjuryProfiler
from fantasy_baseball_manager.services.player_eligibility import PlayerEligibilityService
from fantasy_baseball_manager.services.scenario_generator import generate_pool_scenarios
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


def _players() -> list[Player]:
    """Players with birth dates for age computation."""
    return [
        Player(id=1, name_first="Alpha", name_last="One", birth_date="1992-03-15"),  # age ~33 in 2025 → "old"
        Player(id=2, name_first="Beta", name_last="Two", birth_date="2000-06-01"),  # age ~25 in 2025 → "young"
        Player(id=3, name_first="Gamma", name_last="Three", birth_date="1999-01-10"),  # age ~26 → "young"
        Player(id=4, name_first="Delta", name_last="Four", birth_date="1995-09-20"),  # age ~29 → "young"
        Player(id=5, name_first="Epsilon", name_last="Five", birth_date="1998-12-01"),  # age ~26 → "young"
    ]


def _il_stints_for_player_1() -> list[ILStint]:
    """Player 1 has significant IL history across 2023-2024."""
    return [
        ILStint(player_id=1, season=2023, start_date="2023-05-01", il_type="10-day", days=30),
        ILStint(player_id=1, season=2024, start_date="2024-06-01", il_type="10-day", days=45),
    ]


def _wide_residual_buckets() -> dict[str, ResidualBuckets]:
    """Residual buckets with wide spread for old_injured, narrow for young_healthy."""
    wide = ResidualPercentiles(
        p10=-200.0, p25=-100.0, p50=0.0, p75=50.0, p90=80.0, count=50, std=100.0, mean_offset=0.0
    )
    narrow = ResidualPercentiles(p10=-10.0, p25=-5.0, p50=0.0, p75=5.0, p90=10.0, count=50, std=5.0, mean_offset=0.0)
    fallback = ResidualPercentiles(
        p10=-50.0, p25=-25.0, p50=0.0, p75=25.0, p90=50.0, count=100, std=30.0, mean_offset=0.0
    )
    return {
        "batter": ResidualBuckets(
            buckets={
                "old_injured": wide,
                "young_healthy": narrow,
                "young_injured": narrow,
                "old_healthy": narrow,
                "all": fallback,
            },
            player_type="batter",
            fallback_key="all",
        ),
        "pitcher": ResidualBuckets(
            buckets={"all": fallback},
            player_type="pitcher",
            fallback_key="all",
        ),
    }


def _build_model(
    stints: list[ILStint] | None = None,
    players: list[Player] | None = None,
    residual_buckets: dict[str, ResidualBuckets] | None = None,
) -> tuple[ZarDistributionalModel, FakeValuationRepo]:
    proj_repo = FakeProjectionRepo(_projections())
    pos_repo = FakePositionAppearanceRepo(_appearances())
    val_repo = FakeValuationRepo()
    player_repo = FakePlayerRepo(players or _players())
    eligibility = PlayerEligibilityService(pos_repo)
    il_repo = FakeILStintRepo(stints)
    profiler = InjuryProfiler(player_repo=player_repo, il_stint_repo=il_repo)
    model = ZarDistributionalModel(
        projection_repo=proj_repo,
        position_repo=pos_repo,
        player_repo=player_repo,
        valuation_repo=val_repo,
        eligibility_service=eligibility,
        injury_profiler=profiler,
        scenario_generator=generate_pool_scenarios,
        distributional_zar_runner=run_distributional_zar,
    )
    return model, val_repo


def _build_plain_zar() -> tuple[ZarModel, FakeValuationRepo]:
    proj_repo = FakeProjectionRepo(_projections())
    pos_repo = FakePositionAppearanceRepo(_appearances())
    val_repo = FakeValuationRepo()
    model = ZarModel(
        projection_repo=proj_repo,
        position_repo=pos_repo,
        player_repo=FakePlayerRepo(_players()),
        valuation_repo=val_repo,
        eligibility_service=PlayerEligibilityService(pos_repo),
    )
    return model, val_repo


def _config(residual_buckets: dict[str, ResidualBuckets] | None = None) -> ModelConfig:
    params: dict[str, object] = {
        "league": _league(),
        "projection_system": "steamer",
    }
    if residual_buckets is not None:
        params["_residual_buckets"] = residual_buckets
    return ModelConfig(
        seasons=[2025],
        model_params=params,
        version="1.0",
    )


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestZarDistributionalProtocol:
    def test_is_model(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Model)

    def test_is_predictable(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Predictable)

    def test_name(self) -> None:
        model, _ = _build_model()
        assert model.name == "zar-distributional"

    def test_supported_operations(self) -> None:
        model, _ = _build_model()
        assert model.supported_operations == frozenset({"predict"})

    def test_model_discoverable(self) -> None:
        assert get("zar-distributional") is ZarDistributionalModel


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------


class TestZarDistributionalPredict:
    def test_predict_produces_valuations_with_distributional_system(self) -> None:
        """All persisted valuations should have system='zar-distributional'."""
        model, val_repo = _build_model(stints=_il_stints_for_player_1())
        model.predict(_config(_wide_residual_buckets()))
        assert len(val_repo.upserted) > 0
        for v in val_repo.upserted:
            assert v.system == "zar-distributional"

    def test_predict_returns_all_players(self) -> None:
        """Same player count as plain ZAR."""
        model, _ = _build_model()
        result = model.predict(_config(_wide_residual_buckets()))
        assert len(result.predictions) == 5

    def test_predict_ranked_valuations(self) -> None:
        """Ranks are sequential 1..N."""
        model, _ = _build_model()
        result = model.predict(_config(_wide_residual_buckets()))
        ranks = sorted(p["rank"] for p in result.predictions)
        assert ranks == [1, 2, 3, 4, 5]

    def test_injury_prone_player_valued_lower(self) -> None:
        """Player with IL history + wide PT distribution gets lower value than plain ZAR."""
        # Plain ZAR
        plain_model, plain_val_repo = _build_plain_zar()
        plain_model.predict(
            ModelConfig(
                seasons=[2025],
                model_params={"league": _league(), "projection_system": "steamer"},
                version="1.0",
            )
        )
        plain_values = {v.player_id: v.value for v in plain_val_repo.upserted}

        # Distributional ZAR with wide buckets for old_injured (player 1)
        model, val_repo = _build_model(stints=_il_stints_for_player_1())
        model.predict(_config(_wide_residual_buckets()))
        dist_values = {v.player_id: v.value for v in val_repo.upserted}

        # Player 1 is old (33) + injured → old_injured bucket with wide spread
        # The wide left tail (p10=-200 PA) should pull expected value down
        assert dist_values[1] < plain_values[1]

    def test_stable_player_close_to_zar(self) -> None:
        """Player with no IL history and narrow distribution gets value close to plain ZAR."""
        # Plain ZAR
        plain_model, plain_val_repo = _build_plain_zar()
        plain_model.predict(
            ModelConfig(
                seasons=[2025],
                model_params={"league": _league(), "projection_system": "steamer"},
                version="1.0",
            )
        )
        plain_values = {v.player_id: v.value for v in plain_val_repo.upserted}

        # Distributional ZAR with narrow buckets for young_healthy (player 2)
        model, val_repo = _build_model(stints=[])
        model.predict(_config(_wide_residual_buckets()))
        dist_values = {v.player_id: v.value for v in val_repo.upserted}

        # Player 2 is young + healthy → narrow bucket (±10 PA)
        # Value should be close to plain ZAR (within $2)
        assert abs(dist_values[2] - plain_values[2]) < 2.0

    def test_no_residual_buckets_falls_back_to_point_estimate(self) -> None:
        """When no residual buckets available, values match plain ZAR exactly."""
        # Plain ZAR
        plain_model, plain_val_repo = _build_plain_zar()
        plain_model.predict(
            ModelConfig(
                seasons=[2025],
                model_params={"league": _league(), "projection_system": "steamer"},
                version="1.0",
            )
        )
        plain_values = {v.player_id: v.value for v in plain_val_repo.upserted}

        # Distributional ZAR with no residual buckets (no _residual_buckets, no artifact path)
        model, val_repo = _build_model()
        model.predict(_config())  # no residual_buckets → fallback
        dist_values = {v.player_id: v.value for v in val_repo.upserted}

        assert plain_values == dist_values

    def test_model_name_in_result(self) -> None:
        model, _ = _build_model()
        result = model.predict(_config(_wide_residual_buckets()))
        assert result.model_name == "zar-distributional"

    def test_fallback_system_name(self) -> None:
        """Even in fallback mode, system is zar-distributional."""
        model, val_repo = _build_model()
        model.predict(_config())  # no residual_buckets → fallback
        for v in val_repo.upserted:
            assert v.system == "zar-distributional"
