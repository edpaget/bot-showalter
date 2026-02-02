from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.pipeline.presets import marcel_pipeline
from fantasy_baseball_manager.pipeline.source import PipelineProjectionSource


def _make_player(year: int = 2024, age: int = 28) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id="p1",
        name="Test",
        year=year,
        age=age,
        pa=600,
        ab=540,
        h=160,
        singles=100,
        doubles=30,
        triples=5,
        hr=25,
        bb=50,
        so=120,
        hbp=5,
        sf=3,
        sh=2,
        sb=10,
        cs=3,
        r=80,
        rbi=90,
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


class TestPipelineProjectionSource:
    def test_implements_projection_source(self) -> None:
        league = _make_league()
        ds = FakeDataSource(
            player_batting={2024: [_make_player()]},
            team_batting={2024: [league], 2023: [league], 2022: [league]},
            team_pitching={
                2024: [_make_league_pitching()],
                2023: [_make_league_pitching()],
                2022: [_make_league_pitching()],
            },
        )
        source = PipelineProjectionSource(marcel_pipeline(), ds, 2025)
        batting = source.batting_projections()
        pitching = source.pitching_projections()
        assert len(batting) == 1
        assert isinstance(batting[0], BattingProjection)
        assert isinstance(pitching, list)
