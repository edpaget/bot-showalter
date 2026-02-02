import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
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


class TestMarcelBattingPipeline:
    def test_single_player_three_years(self) -> None:
        """Pipeline returns a BattingProjection with correct metadata."""
        league = _make_league()
        ds = FakeDataSource(
            player_batting={
                2024: [_make_player(year=2024, age=28, pa=600, hr=25)],
                2023: [_make_player(year=2023, age=27, pa=550, hr=20)],
                2022: [_make_player(year=2022, age=26, pa=500, hr=18)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        result = marcel_pipeline().project_batters(ds, 2025)

        assert len(result) == 1
        assert isinstance(result[0], BattingProjection)
        assert result[0].player_id == "p1"
        assert result[0].year == 2025
        assert result[0].age == 29

    def test_projected_pa_three_years(self) -> None:
        """Projected PA = 0.5 * PA_y1 + 0.1 * PA_y2 + 200."""
        league = _make_league()
        ds = FakeDataSource(
            player_batting={
                2024: [_make_player(year=2024, age=28, pa=600, hr=25)],
                2023: [_make_player(year=2023, age=27, pa=550, hr=20)],
                2022: [_make_player(year=2022, age=26, pa=500, hr=18)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        proj = marcel_pipeline().project_batters(ds, 2025)[0]
        # 0.5*600 + 0.1*550 + 200 = 555
        assert proj.pa == pytest.approx(555.0)

    def test_hr_projection_hand_calculated(self) -> None:
        """Verify HR projection with hand-calculated values.

        Player: 25 HR in 600 PA (y1), 20 HR in 550 PA (y2), 18 HR in 500 PA (y3)
        League HR rate: 200/6000 = 0.03333...

        Weighted rate (before rebaseline):
          num = 5*25 + 4*20 + 3*18 + 1200*0.03333 = 125+80+54+40 = 299
          den = 5*600 + 4*550 + 3*500 + 1200 = 3000+2200+1500+1200 = 7900
          rate = 299/7900 = 0.037848...

        Rebaseline: source and target league rates are both 0.03333 (same
        league data for all 3 years), so rebaseline is identity => 0.037848

        Age adj: age 29 => multiplier = 1.0
        PA = 0.5*600 + 0.1*550 + 200 = 555
        HR = 0.037848 * 555 = 21.006
        """
        league = _make_league()
        ds = FakeDataSource(
            player_batting={
                2024: [_make_player(year=2024, age=28, pa=600, hr=25)],
                2023: [_make_player(year=2023, age=27, pa=550, hr=20)],
                2022: [_make_player(year=2022, age=26, pa=500, hr=18)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        proj = marcel_pipeline().project_batters(ds, 2025)[0]
        assert proj.pa == pytest.approx(555.0)
        # 299/7900 * 555 = 21.006
        assert proj.hr == pytest.approx(21.0, abs=0.1)

    def test_single_player_one_year(self) -> None:
        """Player with only 1 year of data should still project."""
        league = _make_league()
        ds = FakeDataSource(
            player_batting={2024: [_make_player(year=2024, age=22, pa=200, hr=5)]},
            team_batting={2024: [league]},
        )

        result = marcel_pipeline().project_batters(ds, 2025)
        assert len(result) == 1
        proj = result[0]
        # PA = 0.5*200 + 0.1*0 + 200 = 300
        assert proj.pa == pytest.approx(300.0)
        assert proj.hr > 0

    def test_multiple_players(self) -> None:
        """Two players should both get projections."""
        league = _make_league()
        p1 = _make_player(player_id="p1", name="Young", year=2024, age=24, pa=600, hr=25)
        p2 = _make_player(player_id="p2", name="Old", year=2024, age=34, pa=500, hr=30)
        ds = FakeDataSource(
            player_batting={2024: [p1, p2]},
            team_batting={2024: [league]},
        )

        result = marcel_pipeline().project_batters(ds, 2025)
        assert len(result) == 2
        ids = {p.player_id for p in result}
        assert ids == {"p1", "p2"}

    def test_age_adjustment_applied(self) -> None:
        """Young player should project higher rates than old player, all else equal."""
        league = _make_league()
        young = _make_player(player_id="young", year=2024, age=24, pa=600, hr=25)
        old = _make_player(player_id="old", year=2024, age=34, pa=600, hr=25)
        ds = FakeDataSource(
            player_batting={2024: [young, old]},
            team_batting={2024: [league]},
        )

        result = marcel_pipeline().project_batters(ds, 2025)
        proj_map = {p.player_id: p for p in result}
        assert proj_map["young"].hr > proj_map["old"].hr


class TestMarcelPitchingPipeline:
    def test_single_starter_three_years(self) -> None:
        """Pipeline returns a PitchingProjection with correct metadata."""
        league = _make_league_pitching()
        ds = FakeDataSource(
            player_pitching={
                2024: [_make_pitcher(year=2024, age=28, ip=180.0)],
                2023: [_make_pitcher(year=2023, age=27, ip=170.0)],
                2022: [_make_pitcher(year=2022, age=26, ip=160.0)],
            },
            team_pitching={2024: [league], 2023: [league], 2022: [league]},
        )

        result = marcel_pipeline().project_pitchers(ds, 2025)
        assert len(result) == 1
        assert isinstance(result[0], PitchingProjection)
        assert result[0].player_id == "sp1"
        assert result[0].year == 2025
        assert result[0].age == 29

    def test_starter_projected_ip(self) -> None:
        """Starter: 0.5*IP_y1 + 0.1*IP_y2 + 60."""
        league = _make_league_pitching()
        ds = FakeDataSource(
            player_pitching={
                2024: [_make_pitcher(year=2024, age=28, ip=180.0, gs=32, g=32)],
                2023: [_make_pitcher(year=2023, age=27, ip=170.0, gs=30, g=30)],
            },
            team_pitching={2024: [league], 2023: [league]},
        )

        proj = marcel_pipeline().project_pitchers(ds, 2025)[0]
        # 0.5*180 + 0.1*170 + 60 = 167
        assert proj.ip == pytest.approx(167.0)

    def test_reliever_projected_ip(self) -> None:
        """Reliever: 0.5*IP_y1 + 0.1*IP_y2 + 25."""
        league = _make_league_pitching()
        ds = FakeDataSource(
            player_pitching={
                2024: [_make_pitcher(player_id="rp1", year=2024, age=28, ip=70.0, g=65, gs=0)],
                2023: [_make_pitcher(player_id="rp1", year=2023, age=27, ip=65.0, g=60, gs=0)],
            },
            team_pitching={2024: [league], 2023: [league]},
        )

        proj = marcel_pipeline().project_pitchers(ds, 2025)[0]
        # 0.5*70 + 0.1*65 + 25 = 66.5
        assert proj.ip == pytest.approx(66.5)

    def test_era_and_whip_computed(self) -> None:
        """ERA and WHIP are derived correctly from projected counting stats."""
        league = _make_league_pitching()
        ds = FakeDataSource(
            player_pitching={2024: [_make_pitcher(year=2024, age=28)]},
            team_pitching={2024: [league]},
        )

        proj = marcel_pipeline().project_pitchers(ds, 2025)[0]
        expected_era = (proj.er / proj.ip) * 9
        assert proj.era == pytest.approx(expected_era)
        expected_whip = (proj.h + proj.bb) / proj.ip
        assert proj.whip == pytest.approx(expected_whip)

    def test_multiple_pitchers(self) -> None:
        """Multiple pitchers should all get projections."""
        league = _make_league_pitching()
        sp = _make_pitcher(player_id="sp1", year=2024, age=28, gs=30, g=30)
        rp = _make_pitcher(player_id="rp1", year=2024, age=30, ip=70.0, g=65, gs=0)
        ds = FakeDataSource(
            player_pitching={2024: [sp, rp]},
            team_pitching={2024: [league]},
        )

        result = marcel_pipeline().project_pitchers(ds, 2025)
        assert len(result) == 2
        ids = {p.player_id for p in result}
        assert ids == {"sp1", "rp1"}

    def test_age_adjustment_applied(self) -> None:
        """Young pitcher should project more strikeouts than old pitcher."""
        league = _make_league_pitching()
        young = _make_pitcher(player_id="young", year=2024, age=24, so=200, ip=180.0)
        old = _make_pitcher(player_id="old", year=2024, age=34, so=200, ip=180.0)
        ds = FakeDataSource(
            player_pitching={2024: [young, old]},
            team_pitching={2024: [league]},
        )

        result = marcel_pipeline().project_pitchers(ds, 2025)
        proj_map = {p.player_id: p for p in result}
        assert proj_map["young"].so > proj_map["old"].so
