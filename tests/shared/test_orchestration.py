from collections.abc import Generator
from typing import Any

import pytest

from fantasy_baseball_manager.context import get_context
from fantasy_baseball_manager.draft.models import RosterConfig, RosterSlot
from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
from fantasy_baseball_manager.result import Ok
from fantasy_baseball_manager.services import ServiceContainer, cli_context, set_container
from fantasy_baseball_manager.shared.orchestration import (
    _apply_pool_replacement,
    build_projections_and_positions,
)

YEARS = [2024, 2023, 2022]


def _make_batter(
    player_id: str = "b1",
    name: str = "Test Hitter",
    year: int = 2024,
    age: int = 28,
    pa: int = 600,
    ab: int = 540,
    h: int = 160,
    singles: int = 100,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 25,
    bb: int = 50,
    so: int = 120,
    hbp: int = 5,
    sf: int = 3,
    sh: int = 2,
    sb: int = 10,
    cs: int = 3,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        pa=pa,
        ab=ab,
        h=h,
        singles=singles,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
        sh=sh,
        sb=sb,
        cs=cs,
        r=80,
        rbi=90,
    )


def _make_pitcher(
    player_id: str = "p1",
    name: str = "Test Pitcher",
    year: int = 2024,
    age: int = 28,
    ip: float = 180.0,
    g: int = 32,
    gs: int = 32,
    er: int = 70,
    h: int = 150,
    bb: int = 50,
    so: int = 200,
    hr: int = 20,
    hbp: int = 5,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        ip=ip,
        g=g,
        gs=gs,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=hr,
        hbp=hbp,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


def _make_league_batting(year: int = 2024) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        pa=6000,
        ab=5400,
        h=1500,
        singles=900,
        doubles=300,
        triples=30,
        hr=200,
        bb=500,
        so=1400,
        hbp=50,
        sf=30,
        sh=20,
        sb=100,
        cs=30,
        r=800,
        rbi=750,
    )


def _make_league_pitching(year: int = 2024) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        ip=1400.0,
        g=500,
        gs=200,
        er=600,
        h=1300,
        bb=450,
        so=1300,
        hr=180,
        hbp=50,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


class FakeDataSource:
    def __init__(
        self,
        player_batting: dict[int, list[BattingSeasonStats]],
        player_pitching: dict[int, list[PitchingSeasonStats]],
        team_batting_stats: dict[int, list[BattingSeasonStats]],
        team_pitching_stats: dict[int, list[PitchingSeasonStats]],
    ) -> None:
        self._player_batting = player_batting
        self._player_pitching = player_pitching
        self._team_batting = team_batting_stats
        self._team_pitching = team_pitching_stats

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._player_batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._player_pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return self._team_batting.get(year, [])

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._team_pitching.get(year, [])


def _build_fake(
    num_batters: int = 3,
    num_pitchers: int = 3,
) -> FakeDataSource:
    batter_configs = [
        ("b1", "Slugger Jones", 40, 5, 600),
        ("b2", "Speedy Smith", 10, 30, 600),
        ("b3", "Average Andy", 20, 15, 600),
    ]
    # (pid, name, so, er, h, g, gs) — gs/g < 0.5 → RP
    pitcher_configs = [
        ("p1", "Ace Adams", 250, 50, 130, 32, 32),
        ("p2", "Bullpen Bob", 150, 70, 180, 60, 0),
        ("p3", "Middle Mike", 180, 60, 155, 32, 32),
    ]

    player_batting: dict[int, list[BattingSeasonStats]] = {}
    team_batting: dict[int, list[BattingSeasonStats]] = {}
    player_pitching: dict[int, list[PitchingSeasonStats]] = {}
    team_pitching: dict[int, list[PitchingSeasonStats]] = {}

    for y in YEARS:
        batters: list[BattingSeasonStats] = []
        for i in range(min(num_batters, len(batter_configs))):
            pid, name, hr, sb, pa = batter_configs[i]
            batters.append(
                _make_batter(
                    player_id=pid,
                    name=name,
                    year=y,
                    age=28 - (2024 - y),
                    hr=hr,
                    sb=sb,
                    pa=pa,
                )
            )
        player_batting[y] = batters
        team_batting[y] = [_make_league_batting(year=y)]

    for y in YEARS:
        pitchers: list[PitchingSeasonStats] = []
        for i in range(min(num_pitchers, len(pitcher_configs))):
            pid, name, so, er, h, g, gs = pitcher_configs[i]
            pitchers.append(
                _make_pitcher(
                    player_id=pid,
                    name=name,
                    year=y,
                    age=28 - (2024 - y),
                    so=so,
                    er=er,
                    h=h,
                    g=g,
                    gs=gs,
                )
            )
        player_pitching[y] = pitchers
        team_pitching[y] = [_make_league_pitching(year=y)]

    return FakeDataSource(
        player_batting=player_batting,
        player_pitching=player_pitching,
        team_batting_stats=team_batting,
        team_pitching_stats=team_pitching,
    )


def _wrap_source(method: Any) -> Any:
    def source(query: Any) -> Ok:
        return Ok(method(get_context().year))

    return source


def _sources_kwargs(ds: FakeDataSource) -> dict[str, Any]:
    return {
        "batting_source": _wrap_source(ds.batting_stats),
        "team_batting_source": _wrap_source(ds.team_batting),
        "pitching_source": _wrap_source(ds.pitching_stats),
        "team_pitching_source": _wrap_source(ds.team_pitching),
    }


def _install_fake(num_batters: int = 3, num_pitchers: int = 3) -> None:
    ds = _build_fake(num_batters=num_batters, num_pitchers=num_pitchers)
    set_container(ServiceContainer(**_sources_kwargs(ds)))


@pytest.fixture(autouse=True)
def reset_container() -> Generator[None]:
    yield
    set_container(None)


class TestApplyPoolReplacementImport:
    def test_importable_from_orchestration(self) -> None:
        assert callable(_apply_pool_replacement)


class TestBuildProjectionsAndPositions:
    def test_returns_values_and_positions(self) -> None:
        _install_fake()
        with cli_context():
            values, positions = build_projections_and_positions("marcel", 2025)
        assert len(values) > 0
        assert len(positions) > 0
        # Should have both batters and pitchers
        batter_values = [v for v in values if v.position_type == "B"]
        pitcher_values = [v for v in values if v.position_type == "P"]
        assert len(batter_values) > 0
        assert len(pitcher_values) > 0

    def test_accepts_roster_config_parameter(self) -> None:
        _install_fake()
        config = RosterConfig(
            slots=(
                RosterSlot(position="C", count=1),
                RosterSlot(position="SP", count=5),
                RosterSlot(position="RP", count=2),
            )
        )
        with cli_context():
            values, positions = build_projections_and_positions("marcel", 2025, roster_config=config)
        assert len(values) > 0
        assert len(positions) > 0

    def test_pitcher_values_adjusted_by_default(self) -> None:
        """Pitcher total_value should differ from raw z-score after VORP adjustment."""
        _install_fake()
        with cli_context():
            # Get raw z-score values (no VORP) by computing z-scores directly
            from fantasy_baseball_manager.config import load_league_settings
            from fantasy_baseball_manager.pipeline.presets import PIPELINES
            from fantasy_baseball_manager.services import get_container
            from fantasy_baseball_manager.valuation.zscore import zscore_pitching

            container = get_container()
            league_settings = load_league_settings()
            pipeline = PIPELINES["marcel"]()
            pitching_projections = pipeline.project_pitchers(
                container.pitching_source, container.team_pitching_source, 2025
            )
            raw_pitching = zscore_pitching(pitching_projections, league_settings.pitching_categories)
            raw_map = {p.player_id: p.total_value for p in raw_pitching}

            # Get VORP-adjusted values
            values, _ = build_projections_and_positions("marcel", 2025)
        pitcher_values = {v.player_id: v.total_value for v in values if v.position_type == "P"}
        # At least one pitcher should have a different value after adjustment
        assert any(
            pitcher_values[pid] != raw_map[pid]
            for pid in pitcher_values
            if pid in raw_map
        )

    def test_batter_values_unchanged_without_positions(self) -> None:
        """Batter values should be identical to raw z-score (no batter positions in orchestration)."""
        _install_fake()
        with cli_context():
            from fantasy_baseball_manager.config import load_league_settings
            from fantasy_baseball_manager.pipeline.presets import PIPELINES
            from fantasy_baseball_manager.services import get_container
            from fantasy_baseball_manager.valuation.zscore import zscore_batting

            container = get_container()
            league_settings = load_league_settings()
            pipeline = PIPELINES["marcel"]()
            batting_projections = pipeline.project_batters(
                container.batting_source, container.team_batting_source, 2025
            )
            raw_batting = zscore_batting(batting_projections, league_settings.batting_categories)
            raw_map = {p.player_id: p.total_value for p in raw_batting}

            values, _ = build_projections_and_positions("marcel", 2025)
        batter_values = {v.player_id: v.total_value for v in values if v.position_type == "B"}
        # All batter values should be identical to raw z-score
        for pid in batter_values:
            if pid in raw_map:
                assert batter_values[pid] == raw_map[pid], f"Batter {pid} value changed unexpectedly"

    def test_custom_roster_config_affects_adjustment(self) -> None:
        """Different roster config should produce different pitcher values."""
        # Config with both SP and RP slots
        sp_rp_config = RosterConfig(
            slots=(
                RosterSlot(position="SP", count=2),
                RosterSlot(position="RP", count=2),
            )
        )
        # Config with only SP slots (no RP)
        sp_only_config = RosterConfig(
            slots=(
                RosterSlot(position="SP", count=4),
            )
        )

        _install_fake()
        with cli_context():
            values_a, _ = build_projections_and_positions("marcel", 2025, roster_config=sp_rp_config)

        pitcher_a = {v.player_id: v.total_value for v in values_a if v.position_type == "P"}

        _install_fake()
        with cli_context():
            values_b, _ = build_projections_and_positions("marcel", 2025, roster_config=sp_only_config)

        pitcher_b = {v.player_id: v.total_value for v in values_b if v.position_type == "P"}

        # At least one pitcher should have a different value with different config
        assert any(
            pitcher_b[pid] != pitcher_a[pid]
            for pid in pitcher_b
            if pid in pitcher_a
        )
