"""Tests for MTL dataset classes."""

from typing import Any, cast

import numpy as np
import torch

from fantasy_baseball_manager.context import init_context, reset_context
from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
from fantasy_baseball_manager.ml.mtl.dataset import (
    BatterTrainingDataCollector,
    MTLDataset,
    PitcherTrainingDataCollector,
)
from fantasy_baseball_manager.ml.mtl.model import BATTER_STATS, PITCHER_STATS
from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
from fantasy_baseball_manager.pipeline.statcast_data import StatcastBatterStats, StatcastPitcherStats
from fantasy_baseball_manager.result import Ok


def _fake_batting_source(data: dict[int, list[BattingSeasonStats]]) -> Any:
    """Create a fake DataSource[BattingSeasonStats] callable."""

    def source(query: Any) -> Ok[list[BattingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        return Ok(data.get(get_context().year, []))

    return source


def _fake_pitching_source(data: dict[int, list[PitchingSeasonStats]]) -> Any:
    """Create a fake DataSource[PitchingSeasonStats] callable."""

    def source(query: Any) -> Ok[list[PitchingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        return Ok(data.get(get_context().year, []))

    return source


class FakeStatcastSource:
    def __init__(
        self,
        batter_stats: dict[int, list[StatcastBatterStats]],
        pitcher_stats: dict[int, list[StatcastPitcherStats]],
    ) -> None:
        self._batter = batter_stats
        self._pitcher = pitcher_stats

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        return self._batter.get(year, [])

    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]:
        return self._pitcher.get(year, [])


class FakeBattedBallSource:
    def __init__(self, stats: dict[int, list[PitcherBattedBallStats]]) -> None:
        self._stats = stats

    def pitcher_batted_ball_stats(self, year: int) -> list[PitcherBattedBallStats]:
        return self._stats.get(year, [])


class FakeSkillDataSource:
    def __init__(
        self,
        batter_stats: dict[int, list[Any]] | None = None,
        pitcher_stats: dict[int, list[Any]] | None = None,
    ) -> None:
        self._batter = batter_stats or {}
        self._pitcher = pitcher_stats or {}

    def batter_skill_stats(self, year: int) -> list[Any]:
        return self._batter.get(year, [])

    def pitcher_skill_stats(self, year: int) -> list[Any]:
        return self._pitcher.get(year, [])


class FakeIdMapper:
    def __init__(self, fg_to_mlbam: dict[str, str]) -> None:
        self._fg_to_mlbam = fg_to_mlbam

    def fangraphs_to_mlbam(self, fg_id: str) -> str | None:
        return self._fg_to_mlbam.get(fg_id)


def _make_batter_actuals(player_id: str, year: int) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=f"Player {player_id}",
        year=year,
        age=28,
        pa=520,
        ab=470,
        h=130,
        singles=85,
        doubles=28,
        triples=4,
        hr=22,
        bb=55,
        so=95,
        hbp=5,
        sf=4,
        sh=1,
        sb=12,
        cs=4,
        r=75,
        rbi=70,
    )


def _make_statcast_batter(mlbam_id: str, year: int) -> StatcastBatterStats:
    return StatcastBatterStats(
        player_id=mlbam_id,
        name=f"Player {mlbam_id}",
        year=year,
        pa=450,
        barrel_rate=0.08,
        hard_hit_rate=0.40,
        xwoba=0.350,
        xba=0.280,
        xslg=0.450,
    )


def _make_statcast_pitcher(mlbam_id: str, year: int) -> StatcastPitcherStats:
    return StatcastPitcherStats(
        player_id=mlbam_id,
        name=f"Player {mlbam_id}",
        year=year,
        pa=500,
        xba=0.250,
        xslg=0.400,
        xwoba=0.320,
        xera=3.80,
        barrel_rate=0.06,
        hard_hit_rate=0.35,
    )


def _make_pitching_actuals(player_id: str, year: int) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=f"Player {player_id}",
        year=year,
        age=28,
        ip=180.0,
        g=30,
        gs=28,
        er=65,
        h=155,
        bb=50,
        so=170,
        hr=20,
        hbp=5,
        w=12,
        sv=0,
        hld=0,
        bs=0,
    )


def _make_batted_ball(player_id: str, year: int) -> PitcherBattedBallStats:
    return PitcherBattedBallStats(
        player_id=player_id,
        name=f"Player {player_id}",
        year=year,
        pa=500,
        gb_pct=0.45,
        fb_pct=0.35,
        ld_pct=0.20,
        iffb_pct=0.10,
    )


class TestMTLDataset:
    def test_init(self) -> None:
        """Test dataset initialization."""
        n_samples = 100
        n_features = 25
        features = np.random.randn(n_samples, n_features).astype(np.float32)
        rates = {stat: np.random.rand(n_samples).astype(np.float32) for stat in BATTER_STATS}

        dataset = MTLDataset(features, rates)

        assert len(dataset) == n_samples

    def test_getitem(self) -> None:
        """Test getting a single item."""
        n_samples = 10
        n_features = 5
        features = np.random.randn(n_samples, n_features).astype(np.float32)
        rates = {"hr": np.random.rand(n_samples).astype(np.float32)}

        dataset = MTLDataset(features, rates)
        item_features, item_rates = dataset[0]

        assert item_features.shape == (n_features,)
        assert "hr" in item_rates
        assert item_rates["hr"].shape == (1,)

    def test_getitem_all_stats(self) -> None:
        """Test that all stats are returned in rates dict."""
        features = np.random.randn(5, 10).astype(np.float32)
        rates = {
            "hr": np.random.rand(5).astype(np.float32),
            "so": np.random.rand(5).astype(np.float32),
            "bb": np.random.rand(5).astype(np.float32),
        }

        dataset = MTLDataset(features, rates)
        _, item_rates = dataset[2]

        assert set(item_rates.keys()) == {"hr", "so", "bb"}

    def test_tensor_types(self) -> None:
        """Test that tensors are float32."""
        features = np.random.randn(5, 10).astype(np.float64)  # Note: float64
        rates = {"hr": np.random.rand(5).astype(np.float64)}

        dataset = MTLDataset(features, rates)
        item_features, item_rates = dataset[0]

        assert item_features.dtype == torch.float32
        assert item_rates["hr"].dtype == torch.float32

    def test_works_with_dataloader(self) -> None:
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        features = np.random.randn(20, 10).astype(np.float32)
        rates = {"hr": np.random.rand(20).astype(np.float32)}

        dataset = MTLDataset(features, rates)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        batch_features, batch_rates = next(iter(loader))

        assert batch_features.shape == (4, 10)
        assert batch_rates["hr"].shape == (4, 1)


class TestBatterTrainingDataCollectorFeatureStore:
    """Tests that BatterTrainingDataCollector delegates to FeatureStore when provided."""

    def setup_method(self) -> None:
        init_context(year=2024)

    def teardown_method(self) -> None:
        reset_context()

    def test_feature_store_used_when_provided(self) -> None:
        """Collect batter training data via FeatureStore path and verify results."""
        from fantasy_baseball_manager.pipeline.feature_store import FeatureStore

        n_players = 40
        players = [f"fg{i}" for i in range(n_players)]
        fg_to_mlbam = {f"fg{i}": f"mlbam{i}" for i in range(n_players)}

        batting_actuals: dict[int, list[BattingSeasonStats]] = {}
        statcast_batter: dict[int, list[StatcastBatterStats]] = {}

        for year in [2022, 2023]:
            batting_actuals[year] = [_make_batter_actuals(p, year) for p in players]
            # Prior year statcast (features come from year - 1)
            batting_actuals[year - 1] = [_make_batter_actuals(p, year - 1) for p in players]
            statcast_batter[year - 1] = [
                _make_statcast_batter(f"mlbam{i}", year - 1) for i in range(n_players)
            ]

        statcast_source = FakeStatcastSource(statcast_batter, {})
        skill_source = FakeSkillDataSource()

        store = FeatureStore(
            statcast_source=statcast_source,
            batted_ball_source=cast("Any", FakeBattedBallSource({})),
            skill_data_source=skill_source,
        )

        collector = BatterTrainingDataCollector(
            batting_source=_fake_batting_source(batting_actuals),
            statcast_source=statcast_source,
            skill_data_source=skill_source,
            id_mapper=cast("Any", FakeIdMapper(fg_to_mlbam)),
            feature_store=store,
        )

        features, rates, feature_names = collector.collect((2022, 2023))

        assert len(features) > 0
        assert features.shape[0] == len(rates["hr"])
        for stat in BATTER_STATS:
            assert stat in rates
            assert len(rates[stat]) == features.shape[0]
        assert len(feature_names) > 0


class TestPitcherTrainingDataCollectorFeatureStore:
    """Tests that PitcherTrainingDataCollector delegates to FeatureStore when provided."""

    def setup_method(self) -> None:
        init_context(year=2024)

    def teardown_method(self) -> None:
        reset_context()

    def test_feature_store_used_when_provided(self) -> None:
        """Collect pitcher training data via FeatureStore path and verify results."""
        from fantasy_baseball_manager.pipeline.feature_store import FeatureStore

        n_players = 40
        players = [f"fg{i}" for i in range(n_players)]
        fg_to_mlbam = {f"fg{i}": f"mlbam{i}" for i in range(n_players)}

        pitching_actuals: dict[int, list[PitchingSeasonStats]] = {}
        statcast_pitcher: dict[int, list[StatcastPitcherStats]] = {}
        batted_ball: dict[int, list[PitcherBattedBallStats]] = {}

        for year in [2022, 2023]:
            pitching_actuals[year] = [_make_pitching_actuals(p, year) for p in players]
            # Prior year data (features come from year - 1)
            pitching_actuals[year - 1] = [_make_pitching_actuals(p, year - 1) for p in players]
            statcast_pitcher[year - 1] = [
                _make_statcast_pitcher(f"mlbam{i}", year - 1) for i in range(n_players)
            ]
            batted_ball[year - 1] = [_make_batted_ball(f"fg{i}", year - 1) for i in range(n_players)]

        statcast_source = FakeStatcastSource({}, statcast_pitcher)
        batted_ball_source = FakeBattedBallSource(batted_ball)
        skill_source = FakeSkillDataSource()

        store = FeatureStore(
            statcast_source=statcast_source,
            batted_ball_source=batted_ball_source,
            skill_data_source=skill_source,
        )

        collector = PitcherTrainingDataCollector(
            pitching_source=_fake_pitching_source(pitching_actuals),
            statcast_source=statcast_source,
            batted_ball_source=batted_ball_source,
            id_mapper=cast("Any", FakeIdMapper(fg_to_mlbam)),
            feature_store=store,
        )

        features, rates, feature_names = collector.collect((2022, 2023))

        assert len(features) > 0
        assert features.shape[0] == len(rates["h"])
        for stat in PITCHER_STATS:
            assert stat in rates
            assert len(rates[stat]) == features.shape[0]
        assert len(feature_names) > 0
