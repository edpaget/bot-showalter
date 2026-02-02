import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.pipeline.stages.stat_specific_rate_computer import (
    StatSpecificRegressionRateComputer,
)


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
    team: str = "NYY",
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
        team=team,
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
    team: str = "NYY",
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
        team=team,
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


class TestStatSpecificRegressionBatting:
    def test_so_rate_weighted_more_toward_actual(self) -> None:
        """SO has regression_pa=200, much less than the 1200 default.

        With lower regression, the player's actual rate carries more weight
        than the league mean. Compare against what a 1200 regression gives.
        """
        league = _make_league()
        player = _make_player(year=2024, age=28, pa=600, so=120)
        ds = FakeDataSource(
            player_batting={2024: [player]},
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        # Stat-specific: SO regression = 200
        computer = StatSpecificRegressionRateComputer()
        rates = computer.compute_batting_rates(ds, 2025, 3)
        so_rate_specific = rates[0].rates["so"]

        # Hand calculation: num = 5*120 + 200*(1400/6000) = 600 + 46.667 = 646.667
        # den = 5*600 + 200 = 3200
        # rate = 646.667/3200 = 0.20208...
        expected = (5 * 120 + 200 * (1400 / 6000)) / (5 * 600 + 200)
        assert so_rate_specific == pytest.approx(expected, abs=1e-6)

    def test_doubles_rate_regresses_more(self) -> None:
        """Doubles has regression_pa=1600, more than the 1200 default.

        With higher regression, the rate moves closer to the league mean.
        """
        league = _make_league()
        player = _make_player(year=2024, age=28, pa=600, doubles=60)
        ds = FakeDataSource(
            player_batting={2024: [player]},
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        computer = StatSpecificRegressionRateComputer()
        rates = computer.compute_batting_rates(ds, 2025, 3)
        doubles_rate = rates[0].rates["doubles"]

        # num = 5*60 + 1600*(300/6000) = 300 + 80 = 380
        # den = 5*600 + 1600 = 4600
        # rate = 380/4600 = 0.08260...
        expected = (5 * 60 + 1600 * (300 / 6000)) / (5 * 600 + 1600)
        assert doubles_rate == pytest.approx(expected, abs=1e-6)

    def test_custom_regression_overrides_defaults(self) -> None:
        league = _make_league()
        player = _make_player(year=2024, age=28, pa=600, hr=25)
        ds = FakeDataSource(
            player_batting={2024: [player]},
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        custom_reg = {"hr": 100}  # Very low regression
        computer = StatSpecificRegressionRateComputer(batting_regression=custom_reg)
        rates = computer.compute_batting_rates(ds, 2025, 3)
        hr_rate = rates[0].rates["hr"]

        # num = 5*25 + 100*(200/6000) = 125 + 3.333 = 128.333
        # den = 5*600 + 100 = 3100
        expected = (5 * 25 + 100 * (200 / 6000)) / (5 * 600 + 100)
        assert hr_rate == pytest.approx(expected, abs=1e-6)

    def test_metadata_contains_team(self) -> None:
        league = _make_league()
        player = _make_player(team="BOS")
        ds = FakeDataSource(
            player_batting={2024: [player]},
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )
        computer = StatSpecificRegressionRateComputer()
        rates = computer.compute_batting_rates(ds, 2025, 3)
        assert rates[0].metadata["team"] == "BOS"


class TestStatSpecificRegressionPitching:
    def test_so_rate_regression(self) -> None:
        """Pitching SO regression is 30 outs (very low), trusting K rate."""
        league = _make_league_pitching()
        pitcher = _make_pitcher(year=2024, ip=180.0, so=200)
        ds = FakeDataSource(
            player_pitching={2024: [pitcher]},
            team_pitching={2024: [league], 2023: [league], 2022: [league]},
        )

        computer = StatSpecificRegressionRateComputer()
        rates = computer.compute_pitching_rates(ds, 2025, 3)
        so_rate = rates[0].rates["so"]

        # outs = 180*3 = 540
        # league_rate = 1400/(1450*3) = 1400/4350
        # num = 3*200 + 30*(1400/4350) = 600 + 9.655 = 609.655
        # den = 3*540 + 30 = 1650
        league_rate = 1400 / (1450 * 3)
        expected = (3 * 200 + 30 * league_rate) / (3 * 540 + 30)
        assert so_rate == pytest.approx(expected, abs=1e-6)

    def test_h_rate_high_regression(self) -> None:
        """Pitching H regression is 200 outs (high), regressing BABIP."""
        league = _make_league_pitching()
        pitcher = _make_pitcher(year=2024, ip=180.0, h=150)
        ds = FakeDataSource(
            player_pitching={2024: [pitcher]},
            team_pitching={2024: [league], 2023: [league], 2022: [league]},
        )

        computer = StatSpecificRegressionRateComputer()
        rates = computer.compute_pitching_rates(ds, 2025, 3)
        h_rate = rates[0].rates["h"]

        league_rate = 1350 / (1450 * 3)
        expected = (3 * 150 + 200 * league_rate) / (3 * 540 + 200)
        assert h_rate == pytest.approx(expected, abs=1e-6)
