from pathlib import Path

from typer.testing import CliRunner

from fantasy_baseball_manager.cli import app
from fantasy_baseball_manager.draft.cli import set_data_source_factory
from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)

runner = CliRunner()

YEARS = [2024, 2023, 2022]


def _make_batter(
    player_id: str = "b1",
    name: str = "Test Hitter",
    year: int = 2024,
    hr: int = 25,
    sb: int = 10,
    pa: int = 600,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=28,
        pa=pa,
        ab=540,
        h=160,
        singles=100,
        doubles=30,
        triples=5,
        hr=hr,
        bb=50,
        so=120,
        hbp=5,
        sf=3,
        sh=2,
        sb=sb,
        cs=3,
        r=80,
        rbi=90,
    )


def _make_pitcher(
    player_id: str = "p1",
    name: str = "Test Pitcher",
    year: int = 2024,
    so: int = 200,
    er: int = 70,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=28,
        ip=180.0,
        g=32,
        gs=32,
        er=er,
        h=150,
        bb=50,
        so=so,
        hr=20,
        hbp=5,
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


def _build_fake(num_batters: int = 8, num_pitchers: int = 8) -> FakeDataSource:
    batter_configs = [
        ("b1", "Slugger Jones", 40, 5),
        ("b2", "Speedy Smith", 10, 30),
        ("b3", "Average Andy", 20, 15),
        ("b4", "Power Pete", 35, 3),
        ("b5", "Contact Carl", 15, 20),
        ("b6", "Bench Bob", 8, 5),
        ("b7", "Backup Bill", 5, 3),
        ("b8", "Filler Fred", 3, 2),
    ]
    pitcher_configs = [
        ("p1", "Ace Adams", 250, 50),
        ("p2", "Bullpen Bob", 150, 70),
        ("p3", "Middle Mike", 180, 60),
        ("p4", "Starter Stan", 220, 55),
        ("p5", "Setup Sam", 130, 75),
        ("p6", "Long Larry", 100, 80),
        ("p7", "Spot Steve", 90, 85),
        ("p8", "Mop Mo", 70, 90),
    ]

    player_batting: dict[int, list[BattingSeasonStats]] = {}
    team_batting: dict[int, list[BattingSeasonStats]] = {}
    player_pitching: dict[int, list[PitchingSeasonStats]] = {}
    team_pitching: dict[int, list[PitchingSeasonStats]] = {}

    for y in YEARS:
        batters = [
            _make_batter(player_id=pid, name=name, year=y, hr=hr, sb=sb)
            for pid, name, hr, sb in batter_configs[:num_batters]
        ]
        player_batting[y] = batters
        team_batting[y] = [_make_league_batting(year=y)]

        pitchers = [
            _make_pitcher(player_id=pid, name=name, year=y, so=so, er=er)
            for pid, name, so, er in pitcher_configs[:num_pitchers]
        ]
        player_pitching[y] = pitchers
        team_pitching[y] = [_make_league_pitching(year=y)]

    return FakeDataSource(
        player_batting=player_batting,
        player_pitching=player_pitching,
        team_batting_stats=team_batting,
        team_pitching_stats=team_pitching,
    )


def _install_fake() -> None:
    ds = _build_fake()
    set_data_source_factory(lambda: ds)


class TestKeeperRankCommand:
    def test_rank_shows_candidates(self) -> None:
        _install_fake()
        result = runner.invoke(
            app, ["keeper", "rank", "2025", "--candidates", "b1,b2,b3", "--user-pick", "1"]
        )
        assert result.exit_code == 0, result.output
        assert "Surplus Value" in result.output or "Surplus" in result.output
        assert "Slugger Jones" in result.output
        assert "Speedy Smith" in result.output
        assert "Average Andy" in result.output

    def test_rank_no_candidates_errors(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["keeper", "rank", "2025"])
        assert result.exit_code == 1

    def test_rank_unknown_candidate_errors(self) -> None:
        _install_fake()
        result = runner.invoke(
            app, ["keeper", "rank", "2025", "--candidates", "nonexistent"]
        )
        assert result.exit_code == 1
        assert "Unknown candidate ID" in result.output

    def test_rank_with_keepers_file(self, tmp_path: Path) -> None:
        _install_fake()
        keepers_file = tmp_path / "keepers.yaml"
        keepers_file.write_text(
            "teams:\n"
            "  1:\n"
            "    keepers: [b4, b5]\n"
            "  2:\n"
            "    keepers: [p1, p2]\n"
        )
        result = runner.invoke(
            app,
            [
                "keeper", "rank", "2025",
                "--candidates", "b1,b2,b3",
                "--keepers", str(keepers_file),
                "--user-pick", "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Slugger Jones" in result.output


class TestKeeperOptimizeCommand:
    def test_optimize_shows_recommendations(self) -> None:
        _install_fake()
        result = runner.invoke(
            app,
            [
                "keeper", "optimize", "2025",
                "--candidates", "b1,b2,b3,b4,b5,b6",
                "--user-pick", "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Optimal Keepers" in result.output
        assert "Total Surplus" in result.output
        assert "All Candidates" in result.output

    def test_optimize_no_candidates_errors(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["keeper", "optimize", "2025"])
        assert result.exit_code == 1

    def test_optimize_with_keepers_file(self, tmp_path: Path) -> None:
        _install_fake()
        keepers_file = tmp_path / "keepers.yaml"
        keepers_file.write_text(
            "teams:\n"
            "  1:\n"
            "    keepers: [p1, p2]\n"
        )
        result = runner.invoke(
            app,
            [
                "keeper", "optimize", "2025",
                "--candidates", "b1,b2,b3,b4,b5",
                "--keepers", str(keepers_file),
                "--user-pick", "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Optimal Keepers" in result.output
