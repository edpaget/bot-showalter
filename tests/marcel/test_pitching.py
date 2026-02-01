import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.marcel.pitching import project_pitchers


class FakePitchingDataSource:
    def __init__(
        self,
        player_stats: dict[int, list[PitchingSeasonStats]],
        team_stats: dict[int, list[PitchingSeasonStats]],
    ) -> None:
        self._player_stats = player_stats
        self._team_stats = team_stats

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return []

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._player_stats.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return []

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._team_stats.get(year, [])


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
    )


def _make_league_pitching(
    year: int = 2024,
    ip: float = 1450.0,
    g: int = 500,
    gs: int = 162,
    er: int = 650,
    h: int = 1350,
    bb: int = 500,
    so: int = 1400,
    hr: int = 180,
    hbp: int = 60,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        ip=ip,
        g=g,
        gs=gs,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=hr,
        hbp=hbp,
    )


class TestProjectPitchers:
    def test_returns_projections(self) -> None:
        pitcher_y1 = _make_pitcher(year=2024, age=28)
        pitcher_y2 = _make_pitcher(year=2023, age=27, ip=170.0)
        pitcher_y3 = _make_pitcher(year=2022, age=26, ip=160.0)
        league = _make_league_pitching()

        ds = FakePitchingDataSource(
            player_stats={
                2024: [pitcher_y1],
                2023: [pitcher_y2],
                2022: [pitcher_y3],
            },
            team_stats={
                2024: [league],
                2023: [league],
                2022: [league],
            },
        )
        projections = project_pitchers(ds, 2025)
        assert len(projections) == 1
        assert isinstance(projections[0], PitchingProjection)
        assert projections[0].player_id == "sp1"
        assert projections[0].year == 2025
        assert projections[0].age == 29

    def test_starter_projected_ip(self) -> None:
        """Starter: 0.5*IP_y1 + 0.1*IP_y2 + 60."""
        pitcher_y1 = _make_pitcher(year=2024, age=28, ip=180.0, gs=32, g=32)
        pitcher_y2 = _make_pitcher(year=2023, age=27, ip=170.0, gs=30, g=30)
        league = _make_league_pitching()

        ds = FakePitchingDataSource(
            player_stats={2024: [pitcher_y1], 2023: [pitcher_y2]},
            team_stats={2024: [league], 2023: [league]},
        )
        proj = project_pitchers(ds, 2025)[0]
        # 0.5*180 + 0.1*170 + 60 = 90 + 17 + 60 = 167
        assert proj.ip == pytest.approx(167.0)

    def test_reliever_projected_ip(self) -> None:
        """Reliever: 0.5*IP_y1 + 0.1*IP_y2 + 25."""
        reliever_y1 = _make_pitcher(player_id="rp1", year=2024, age=28, ip=70.0, g=65, gs=0)
        reliever_y2 = _make_pitcher(player_id="rp1", year=2023, age=27, ip=65.0, g=60, gs=0)
        league = _make_league_pitching()

        ds = FakePitchingDataSource(
            player_stats={2024: [reliever_y1], 2023: [reliever_y2]},
            team_stats={2024: [league], 2023: [league]},
        )
        proj = project_pitchers(ds, 2025)[0]
        # 0.5*70 + 0.1*65 + 25 = 35 + 6.5 + 25 = 66.5
        assert proj.ip == pytest.approx(66.5)

    def test_era_and_whip_computed(self) -> None:
        pitcher_y1 = _make_pitcher(year=2024, age=28)
        league = _make_league_pitching()

        ds = FakePitchingDataSource(
            player_stats={2024: [pitcher_y1]},
            team_stats={2024: [league]},
        )
        proj = project_pitchers(ds, 2025)[0]
        # ERA = (ER / IP) * 9
        expected_era = (proj.er / proj.ip) * 9
        assert proj.era == pytest.approx(expected_era)
        # WHIP = (H + BB) / IP
        expected_whip = (proj.h + proj.bb) / proj.ip
        assert proj.whip == pytest.approx(expected_whip)

    def test_multiple_pitchers(self) -> None:
        sp = _make_pitcher(player_id="sp1", year=2024, age=28, gs=30, g=30)
        rp = _make_pitcher(player_id="rp1", year=2024, age=30, ip=70.0, g=65, gs=0)
        league = _make_league_pitching()

        ds = FakePitchingDataSource(
            player_stats={2024: [sp, rp]},
            team_stats={2024: [league]},
        )
        projections = project_pitchers(ds, 2025)
        assert len(projections) == 2
        ids = {p.player_id for p in projections}
        assert ids == {"sp1", "rp1"}

    def test_age_adjustment_applied(self) -> None:
        young = _make_pitcher(player_id="young", year=2024, age=24, so=200, ip=180.0)
        old = _make_pitcher(player_id="old", year=2024, age=34, so=200, ip=180.0)
        league = _make_league_pitching()

        ds = FakePitchingDataSource(
            player_stats={2024: [young, old]},
            team_stats={2024: [league]},
        )
        projections = project_pitchers(ds, 2025)
        proj_map = {p.player_id: p for p in projections}
        # Young pitcher gets SO boost (more K's is good)
        assert proj_map["young"].so > proj_map["old"].so

    def test_pitcher_missing_older_years(self) -> None:
        pitcher_y1 = _make_pitcher(year=2024, age=22, ip=100.0)
        league = _make_league_pitching()

        ds = FakePitchingDataSource(
            player_stats={2024: [pitcher_y1]},
            team_stats={2024: [league]},
        )
        projections = project_pitchers(ds, 2025)
        assert len(projections) == 1
        proj = projections[0]
        assert proj.ip > 0
        assert proj.so > 0
