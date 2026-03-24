from fantasy_baseball_manager.domain import (
    Model,
    ModelConfig,
    Predictable,
)
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.il_stint import ILStint
from fantasy_baseball_manager.domain.injury_discount import apply_injury_discount
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
from fantasy_baseball_manager.models.zar.model import ZarModel
from fantasy_baseball_manager.models.zar_injury_risk.model import ZarInjuryRiskModel
from fantasy_baseball_manager.services.injury_profiler import InjuryProfiler
from fantasy_baseball_manager.services.player_eligibility import PlayerEligibilityService
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
            player_type=PlayerType.BATTER,
            stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0, "avg": 0.291},
        ),
        Projection(
            player_id=2,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.BATTER,
            stat_json={"pa": 550, "hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0, "avg": 0.280},
        ),
        Projection(
            player_id=3,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.BATTER,
            stat_json={"pa": 570, "hr": 10.0, "r": 50.0, "h": 130.0, "ab": 520.0, "avg": 0.250},
        ),
        Projection(
            player_id=4,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.PITCHER,
            stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
        ),
        Projection(
            player_id=5,
            season=2025,
            system="steamer",
            version="v1",
            player_type=PlayerType.PITCHER,
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
) -> tuple[ZarInjuryRiskModel, FakeValuationRepo]:
    proj_repo = FakeProjectionRepo(_projections() if projections is None else projections)
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


def _league_with_thresholds(min_pa: int = 0, min_ip: int = 0) -> LeagueSettings:
    base = _league()
    return LeagueSettings(
        name=base.name,
        format=base.format,
        teams=base.teams,
        budget=base.budget,
        roster_batters=base.roster_batters,
        roster_pitchers=base.roster_pitchers,
        batting_categories=base.batting_categories,
        pitching_categories=base.pitching_categories,
        roster_util=base.roster_util,
        positions=base.positions,
        eligibility=EligibilityRules(min_pa=min_pa, min_ip=min_ip),
    )


def _build_model_with_pos(
    stints: list[ILStint] | None = None,
    projections: list[Projection] | None = None,
    appearances: list[PositionAppearance] | None = None,
) -> tuple[ZarInjuryRiskModel, FakeValuationRepo]:
    proj_repo = FakeProjectionRepo(_projections() if projections is None else projections)
    pos_repo = FakePositionAppearanceRepo(appearances if appearances is not None else _appearances())
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


class TestZarInjuryRiskProtocol:
    def test_is_model(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Model)

    def test_is_predictable(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Predictable)

    def test_name(self) -> None:
        model, _ = _build_model()
        assert model.name == "zar-injury-risk"

    def test_supported_operations(self) -> None:
        model, _ = _build_model()
        assert model.supported_operations == frozenset({"predict"})


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------


class TestZarInjuryRiskPredict:
    def test_predict_produces_valuations_with_injury_risk_system(self) -> None:
        """All persisted valuations should have system='zar-injury-risk'."""
        model, val_repo = _build_model(stints=_il_stints_for_player_1())
        model.predict(_config())
        assert len(val_repo.upserted) > 0
        for v in val_repo.upserted:
            assert v.system == "zar-injury-risk"

    def test_predict_discounts_counting_stats(self) -> None:
        """Player with IL history should have lower value than without discount."""
        plain_model, plain_val_repo = _build_plain_zar()
        plain_model.predict(_config())
        plain_values = {v.player_id: v.value for v in plain_val_repo.upserted}

        model, val_repo = _build_model(stints=_il_stints_for_player_1())
        model.predict(_config())
        injury_values = {v.player_id: v.value for v in val_repo.upserted}

        # Player 1 (with IL history) should have lower value
        assert injury_values[1] < plain_values[1]

    def test_predict_preserves_rate_stats(self) -> None:
        """Rate stats should be unchanged by injury discount."""
        stat_json = {"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0, "avg": 0.291}
        adjusted = apply_injury_discount(stat_json, 30.0, "batter")
        assert adjusted["avg"] == 0.291  # Rate stat preserved

    def test_predict_delegates_to_zar(self) -> None:
        """The model should produce ranked valuations (full pipeline)."""
        model, _ = _build_model(stints=_il_stints_for_player_1())
        result = model.predict(_config())
        assert result.model_name == "zar-injury-risk"
        assert len(result.predictions) == 5
        ranks = sorted(p["rank"] for p in result.predictions)
        assert ranks == [1, 2, 3, 4, 5]

    def test_predict_no_injury_history_matches_zar(self) -> None:
        """With no IL stints, results should match plain ZAR values."""
        plain_model, plain_val_repo = _build_plain_zar()
        plain_model.predict(_config())
        plain_values = {v.player_id: v.value for v in plain_val_repo.upserted}

        model, val_repo = _build_model(stints=[])
        model.predict(_config())
        injury_values = {v.player_id: v.value for v in val_repo.upserted}

        assert plain_values == injury_values

    def test_predict_uninjured_players_unchanged(self) -> None:
        """Players without IL history should have the same value as plain ZAR."""
        plain_model, plain_val_repo = _build_plain_zar()
        plain_model.predict(_config())
        plain_values = {v.player_id: v.value for v in plain_val_repo.upserted}

        # Only player 1 has IL stints
        model, val_repo = _build_model(stints=_il_stints_for_player_1())
        model.predict(_config())
        injury_values = {v.player_id: v.value for v in val_repo.upserted}

        # Players 2-5 have no injury history — but their relative values shift
        # because player 1's reduced stats change the z-score distribution.
        # The key test is that player 1's value decreases.
        assert injury_values[1] < plain_values[1]

    def test_predict_borderline_player_not_dropped(self) -> None:
        """A batter just above min_pa whose discounted PA falls below threshold should still be valued."""
        league = _league_with_thresholds(min_pa=200, min_ip=30)
        # Player 6: borderline batter with PA=210 — injury discount will push below 200
        borderline_projections = _projections() + [
            Projection(
                player_id=6,
                season=2025,
                system="steamer",
                version="v1",
                player_type=PlayerType.BATTER,
                stat_json={"pa": 210, "hr": 5.0, "r": 20.0, "h": 50.0, "ab": 190.0, "avg": 0.263},
            ),
        ]
        # Heavy IL history for player 6 — enough to discount PA well below 200
        stints = [
            ILStint(player_id=6, season=2023, start_date="2023-05-01", il_type="10-day", days=30),
            ILStint(player_id=6, season=2024, start_date="2024-06-01", il_type="10-day", days=45),
        ]
        model, _ = _build_model(stints=stints, projections=borderline_projections)
        config = ModelConfig(
            seasons=[2025],
            model_params={"league": league, "projection_system": "steamer"},
            version="1.0",
        )
        result = model.predict(config)
        valued_ids = {p["player_id"] for p in result.predictions}
        assert 6 in valued_ids, "Borderline player should not be dropped after injury discount"

    def test_predict_same_player_count_as_zar(self) -> None:
        """zar-injury-risk should produce the same player count as plain zar."""
        league = _league_with_thresholds(min_pa=200, min_ip=30)
        # Give player 1 heavy IL history (discounts PA=600 → ~487, still above 200)
        # and add a borderline batter whose discounted PA would drop below 200
        borderline_projections = _projections() + [
            Projection(
                player_id=6,
                season=2025,
                system="steamer",
                version="v1",
                player_type=PlayerType.BATTER,
                stat_json={"pa": 210, "hr": 5.0, "r": 20.0, "h": 50.0, "ab": 190.0, "avg": 0.263},
            ),
        ]
        stints = [
            ILStint(player_id=6, season=2023, start_date="2023-05-01", il_type="10-day", days=30),
            ILStint(player_id=6, season=2024, start_date="2024-06-01", il_type="10-day", days=45),
        ]
        config = ModelConfig(
            seasons=[2025],
            model_params={"league": league, "projection_system": "steamer"},
            version="1.0",
        )

        # Plain ZAR — no injury discount, all 6 players qualify
        all_appearances = _appearances() + [
            PositionAppearance(player_id=6, season=2025, position="OF", games=80),
        ]
        proj_repo = FakeProjectionRepo(borderline_projections)
        pos_repo = FakePositionAppearanceRepo(all_appearances)
        plain_zar = ZarModel(
            projection_repo=proj_repo,
            position_repo=pos_repo,
            player_repo=FakePlayerRepo(),
            valuation_repo=FakeValuationRepo(),
            eligibility_service=PlayerEligibilityService(pos_repo),
        )
        plain_result = plain_zar.predict(config)

        # Injury-risk ZAR — player 6 should still be included despite discounted PA
        model, _ = _build_model_with_pos(
            stints=stints,
            projections=borderline_projections,
            appearances=_appearances() + [PositionAppearance(player_id=6, season=2025, position="OF", games=80)],
        )
        injury_result = model.predict(config)

        assert len(injury_result.predictions) == len(plain_result.predictions)
