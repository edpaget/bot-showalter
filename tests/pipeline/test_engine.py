import pytest

from fantasy_baseball_manager.marcel.batting import project_batters
from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.marcel.pitching import project_pitchers
from fantasy_baseball_manager.pipeline.presets import marcel_pipeline


def _make_player(
    player_id: str = "p1",
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
    r: int = 80,
    rbi: int = 90,
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
        r=r,
        rbi=rbi,
    )


def _make_league(year: int = 2024) -> BattingSeasonStats:
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


def _make_pitcher(
    player_id: str = "sp1",
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


def _make_league_pitching(year: int = 2024) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        ip=1450.0,
        g=500,
        gs=162,
        er=650,
        h=1350,
        bb=500,
        so=1400,
        hr=180,
        hbp=60,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


class FakeDataSource:
    def __init__(
        self,
        player_batting: dict[int, list[BattingSeasonStats]] | None = None,
        team_batting: dict[int, list[BattingSeasonStats]] | None = None,
        player_pitching: dict[int, list[PitchingSeasonStats]] | None = None,
        team_pitching: dict[int, list[PitchingSeasonStats]] | None = None,
    ) -> None:
        self._player_batting = player_batting or {}
        self._team_batting = team_batting or {}
        self._player_pitching = player_pitching or {}
        self._team_pitching = team_pitching or {}

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._player_batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._player_pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return self._team_batting.get(year, [])

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._team_pitching.get(year, [])


def _assert_batting_equal(pipeline: BattingProjection, monolith: BattingProjection) -> None:
    assert pipeline.player_id == monolith.player_id
    assert pipeline.name == monolith.name
    assert pipeline.year == monolith.year
    assert pipeline.age == monolith.age
    assert pipeline.pa == pytest.approx(monolith.pa)
    assert pipeline.ab == pytest.approx(monolith.ab)
    assert pipeline.h == pytest.approx(monolith.h)
    assert pipeline.singles == pytest.approx(monolith.singles)
    assert pipeline.doubles == pytest.approx(monolith.doubles)
    assert pipeline.triples == pytest.approx(monolith.triples)
    assert pipeline.hr == pytest.approx(monolith.hr)
    assert pipeline.bb == pytest.approx(monolith.bb)
    assert pipeline.so == pytest.approx(monolith.so)
    assert pipeline.hbp == pytest.approx(monolith.hbp)
    assert pipeline.sf == pytest.approx(monolith.sf)
    assert pipeline.sh == pytest.approx(monolith.sh)
    assert pipeline.sb == pytest.approx(monolith.sb)
    assert pipeline.cs == pytest.approx(monolith.cs)
    assert pipeline.r == pytest.approx(monolith.r)
    assert pipeline.rbi == pytest.approx(monolith.rbi)


def _assert_pitching_equal(pipeline: PitchingProjection, monolith: PitchingProjection) -> None:
    assert pipeline.player_id == monolith.player_id
    assert pipeline.name == monolith.name
    assert pipeline.year == monolith.year
    assert pipeline.age == monolith.age
    assert pipeline.ip == pytest.approx(monolith.ip)
    assert pipeline.g == pytest.approx(monolith.g)
    assert pipeline.gs == pytest.approx(monolith.gs)
    assert pipeline.er == pytest.approx(monolith.er)
    assert pipeline.h == pytest.approx(monolith.h)
    assert pipeline.bb == pytest.approx(monolith.bb)
    assert pipeline.so == pytest.approx(monolith.so)
    assert pipeline.hr == pytest.approx(monolith.hr)
    assert pipeline.hbp == pytest.approx(monolith.hbp)
    assert pipeline.era == pytest.approx(monolith.era)
    assert pipeline.whip == pytest.approx(monolith.whip)
    assert pipeline.w == pytest.approx(monolith.w)
    assert pipeline.nsvh == pytest.approx(monolith.nsvh)


class TestMarcelBattingEquivalence:
    def test_single_player_three_years(self) -> None:
        """Pipeline produces identical output to monolithic for 3-year batter."""
        league = _make_league()
        ds = FakeDataSource(
            player_batting={
                2024: [_make_player(year=2024, age=28, pa=600, hr=25)],
                2023: [_make_player(year=2023, age=27, pa=550, hr=20)],
                2022: [_make_player(year=2022, age=26, pa=500, hr=18)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        monolith = project_batters(ds, 2025)
        pipeline = marcel_pipeline().project_batters(ds, 2025)

        assert len(pipeline) == len(monolith) == 1
        _assert_batting_equal(pipeline[0], monolith[0])

    def test_single_player_one_year(self) -> None:
        """Pipeline produces identical output for player with only 1 year."""
        league = _make_league()
        ds = FakeDataSource(
            player_batting={2024: [_make_player(year=2024, age=22, pa=200, hr=5)]},
            team_batting={2024: [league]},
        )

        monolith = project_batters(ds, 2025)
        pipeline = marcel_pipeline().project_batters(ds, 2025)

        assert len(pipeline) == len(monolith) == 1
        _assert_batting_equal(pipeline[0], monolith[0])

    def test_multiple_players(self) -> None:
        """Pipeline produces identical output for multiple batters."""
        league = _make_league()
        p1 = _make_player(player_id="p1", name="Young", year=2024, age=24, pa=600, hr=25)
        p2 = _make_player(player_id="p2", name="Old", year=2024, age=34, pa=500, hr=30)
        ds = FakeDataSource(
            player_batting={2024: [p1, p2]},
            team_batting={2024: [league]},
        )

        monolith = project_batters(ds, 2025)
        pipeline = marcel_pipeline().project_batters(ds, 2025)

        monolith_map = {p.player_id: p for p in monolith}
        pipeline_map = {p.player_id: p for p in pipeline}

        for pid in monolith_map:
            _assert_batting_equal(pipeline_map[pid], monolith_map[pid])


class TestMarcelPitchingEquivalence:
    def test_starter_three_years(self) -> None:
        """Pipeline produces identical output for 3-year starter."""
        league = _make_league_pitching()
        ds = FakeDataSource(
            player_pitching={
                2024: [_make_pitcher(year=2024, age=28, ip=180.0)],
                2023: [_make_pitcher(year=2023, age=27, ip=170.0)],
                2022: [_make_pitcher(year=2022, age=26, ip=160.0)],
            },
            team_pitching={2024: [league], 2023: [league], 2022: [league]},
        )

        monolith = project_pitchers(ds, 2025)
        pipeline = marcel_pipeline().project_pitchers(ds, 2025)

        assert len(pipeline) == len(monolith) == 1
        _assert_pitching_equal(pipeline[0], monolith[0])

    def test_reliever(self) -> None:
        """Pipeline produces identical output for reliever."""
        league = _make_league_pitching()
        ds = FakeDataSource(
            player_pitching={
                2024: [_make_pitcher(player_id="rp1", year=2024, age=28, ip=70.0, g=65, gs=0)],
                2023: [_make_pitcher(player_id="rp1", year=2023, age=27, ip=65.0, g=60, gs=0)],
            },
            team_pitching={2024: [league], 2023: [league]},
        )

        monolith = project_pitchers(ds, 2025)
        pipeline = marcel_pipeline().project_pitchers(ds, 2025)

        assert len(pipeline) == len(monolith) == 1
        _assert_pitching_equal(pipeline[0], monolith[0])

    def test_mixed_starter_reliever(self) -> None:
        """Pipeline produces identical output for mixed starter + reliever."""
        league = _make_league_pitching()
        sp = _make_pitcher(player_id="sp1", year=2024, age=28, gs=30, g=30)
        rp = _make_pitcher(player_id="rp1", year=2024, age=30, ip=70.0, g=65, gs=0)
        ds = FakeDataSource(
            player_pitching={2024: [sp, rp]},
            team_pitching={2024: [league]},
        )

        monolith = project_pitchers(ds, 2025)
        pipeline = marcel_pipeline().project_pitchers(ds, 2025)

        monolith_map = {p.player_id: p for p in monolith}
        pipeline_map = {p.player_id: p for p in pipeline}

        for pid in monolith_map:
            _assert_pitching_equal(pipeline_map[pid], monolith_map[pid])
