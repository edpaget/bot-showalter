from typing import TYPE_CHECKING, Any

import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.marcel.weights import weighted_rate
from fantasy_baseball_manager.pipeline.stages.platoon_rate_computer import (
    PlatoonRateComputer,
)
from fantasy_baseball_manager.pipeline.stages.split_regression_constants import (
    BATTING_SPLIT_REGRESSION_PA,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates
from fantasy_baseball_manager.result import Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.context import Context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batting(
    player_id: str = "p1",
    name: str = "Test Hitter",
    year: int = 2024,
    age: int = 28,
    pa: int = 400,
    ab: int = 360,
    h: int = 100,
    singles: int = 60,
    doubles: int = 20,
    triples: int = 3,
    hr: int = 17,
    bb: int = 35,
    so: int = 80,
    hbp: int = 3,
    sf: int = 2,
    sh: int = 0,
    sb: int = 8,
    cs: int = 2,
    r: int = 55,
    rbi: int = 60,
    team: str = "",
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


def _make_batting_source(
    data: dict[int, list[BattingSeasonStats]],
) -> Any:
    """Create a fake DataSource[BattingSeasonStats] that reads year from context."""

    def source(_query: object) -> Ok[list[BattingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        year = get_context().year
        return Ok(data.get(year, []))

    return source


def _make_pitching_source(
    data: dict[int, list[PitchingSeasonStats]],
) -> Any:
    """Create a fake DataSource[PitchingSeasonStats] that reads year from context."""

    def source(_query: object) -> Ok[list[PitchingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        year = get_context().year
        return Ok(data.get(year, []))

    return source


class FakeSplitSource:
    def __init__(
        self,
        vs_lhp: dict[int, list[BattingSeasonStats]] | None = None,
        vs_rhp: dict[int, list[BattingSeasonStats]] | None = None,
    ) -> None:
        self._vs_lhp = vs_lhp or {}
        self._vs_rhp = vs_rhp or {}

    def batting_stats_vs_lhp(self, year: int) -> list[BattingSeasonStats]:
        return self._vs_lhp.get(year, [])

    def batting_stats_vs_rhp(self, year: int) -> list[BattingSeasonStats]:
        return self._vs_rhp.get(year, [])


class FakePitchingDelegate:
    """Fake RateComputer that records calls and returns canned results."""

    def __init__(self, pitching_result: list[PlayerRates] | None = None) -> None:
        self._pitching_result = pitching_result or []
        self.pitching_calls: list[tuple[object, object, int, int]] = []

    def compute_batting_rates(
        self,
        batting_source: object,
        team_batting_source: object,
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        return []

    def compute_pitching_rates(
        self,
        pitching_source: object,
        team_pitching_source: object,
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        self.pitching_calls.append((pitching_source, team_pitching_source, year, years_back))
        return self._pitching_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlatoonRateComputerBlending:
    """Blended rate is weighted average of split rates."""

    def _setup(
        self,
        test_context: "Context",
        vs_lhp_hr: int = 10,
        vs_rhp_hr: int = 20,
        vs_lhp_pa: int = 150,
        vs_rhp_pa: int = 450,
        pct_vs_rhp: float = 0.72,
        pct_vs_lhp: float = 0.28,
    ) -> list[PlayerRates]:
        lhp_stats = _make_batting(player_id="p1", year=2023, pa=vs_lhp_pa, hr=vs_lhp_hr)
        rhp_stats = _make_batting(player_id="p1", year=2023, pa=vs_rhp_pa, hr=vs_rhp_hr)
        league = _make_league_batting(year=2023)

        split_source = FakeSplitSource(
            vs_lhp={2023: [lhp_stats]},
            vs_rhp={2023: [rhp_stats]},
        )
        delegate = FakePitchingDelegate()
        team_batting_source = _make_batting_source({2023: [league]})
        # batting_source not used by PlatoonRateComputer but required by protocol
        batting_source = _make_batting_source({})

        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
            pct_vs_rhp=pct_vs_rhp,
            pct_vs_lhp=pct_vs_lhp,
        )
        return computer.compute_batting_rates(
            batting_source,
            team_batting_source,
            2024,
            1,
        )

    def test_produces_one_result(self, test_context: "Context") -> None:
        result = self._setup(test_context)
        assert len(result) == 1

    def test_blended_rate_is_weighted_average(self, test_context: "Context") -> None:
        result = self._setup(test_context)
        player = result[0]
        rates_vs_lhp = player.metadata["rates_vs_lhp"]
        rates_vs_rhp = player.metadata["rates_vs_rhp"]
        expected_hr = 0.28 * rates_vs_lhp["hr"] + 0.72 * rates_vs_rhp["hr"]
        assert player.rates["hr"] == pytest.approx(expected_hr, rel=1e-6)

    def test_metadata_contains_split_rates(self, test_context: "Context") -> None:
        result = self._setup(test_context)
        player = result[0]
        assert "rates_vs_lhp" in player.metadata
        assert "rates_vs_rhp" in player.metadata
        assert isinstance(player.metadata["rates_vs_lhp"], dict)
        assert isinstance(player.metadata["rates_vs_rhp"], dict)

    def test_metadata_contains_matchup_pcts(self, test_context: "Context") -> None:
        result = self._setup(test_context)
        player = result[0]
        assert player.metadata["pct_vs_rhp"] == 0.72
        assert player.metadata["pct_vs_lhp"] == 0.28

    def test_metadata_contains_standard_keys(self, test_context: "Context") -> None:
        result = self._setup(test_context)
        player = result[0]
        assert "pa_per_year" in player.metadata
        assert "avg_league_rates" in player.metadata
        assert "target_rates" in player.metadata
        assert "team" in player.metadata

    def test_custom_matchup_frequency(self, test_context: "Context") -> None:
        result = self._setup(test_context, pct_vs_rhp=0.60, pct_vs_lhp=0.40)
        player = result[0]
        rates_vs_lhp = player.metadata["rates_vs_lhp"]
        rates_vs_rhp = player.metadata["rates_vs_rhp"]
        expected_hr = 0.40 * rates_vs_lhp["hr"] + 0.60 * rates_vs_rhp["hr"]
        assert player.rates["hr"] == pytest.approx(expected_hr, rel=1e-6)
        assert player.metadata["pct_vs_rhp"] == 0.60
        assert player.metadata["pct_vs_lhp"] == 0.40


class TestPlatoonSplitRegression:
    """Split regression uses doubled constants."""

    def test_uses_doubled_regression_by_default(self, test_context: "Context") -> None:
        lhp_stats = _make_batting(player_id="p1", year=2023, pa=150, hr=5)
        rhp_stats = _make_batting(player_id="p1", year=2023, pa=450, hr=20)
        league = _make_league_batting(year=2023)

        split_source = FakeSplitSource(
            vs_lhp={2023: [lhp_stats]},
            vs_rhp={2023: [rhp_stats]},
        )
        delegate = FakePitchingDelegate()
        batting_source = _make_batting_source({})
        team_batting_source = _make_batting_source({2023: [league]})

        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
        )
        result = computer.compute_batting_rates(
            batting_source,
            team_batting_source,
            2024,
            1,
        )
        player = result[0]

        # Manually compute expected rate vs RHP with doubled regression
        league_hr_rate = 200 / 6000
        regression_pa = BATTING_SPLIT_REGRESSION_PA["hr"]
        expected_vs_rhp = weighted_rate(
            stats=[20.0],
            opportunities=[450.0],
            weights=[5],
            league_rate=league_hr_rate,
            regression_pa=regression_pa,
        )
        assert player.metadata["rates_vs_rhp"]["hr"] == pytest.approx(expected_vs_rhp, rel=1e-6)

    def test_custom_regression_overrides_default(self, test_context: "Context") -> None:
        lhp_stats = _make_batting(player_id="p1", year=2023, pa=150, hr=5)
        rhp_stats = _make_batting(player_id="p1", year=2023, pa=450, hr=20)
        league = _make_league_batting(year=2023)

        split_source = FakeSplitSource(
            vs_lhp={2023: [lhp_stats]},
            vs_rhp={2023: [rhp_stats]},
        )
        delegate = FakePitchingDelegate()
        batting_source = _make_batting_source({})
        team_batting_source = _make_batting_source({2023: [league]})

        custom_regression = {"hr": 100.0}
        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
            batting_regression=custom_regression,
        )
        result = computer.compute_batting_rates(
            batting_source,
            team_batting_source,
            2024,
            1,
        )
        player = result[0]

        league_hr_rate = 200 / 6000
        expected_vs_rhp = weighted_rate(
            stats=[20.0],
            opportunities=[450.0],
            weights=[5],
            league_rate=league_hr_rate,
            regression_pa=100.0,
        )
        assert player.metadata["rates_vs_rhp"]["hr"] == pytest.approx(expected_vs_rhp, rel=1e-6)


class TestPlatoonMissingSplit:
    """Player missing from one split regresses to league average for that split."""

    def test_player_in_rhp_only(self, test_context: "Context") -> None:
        rhp_stats = _make_batting(player_id="p1", year=2023, pa=450, hr=20)
        league = _make_league_batting(year=2023)

        split_source = FakeSplitSource(
            vs_lhp={2023: []},  # p1 missing from LHP split
            vs_rhp={2023: [rhp_stats]},
        )
        delegate = FakePitchingDelegate()
        batting_source = _make_batting_source({})
        team_batting_source = _make_batting_source({2023: [league]})

        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
        )
        result = computer.compute_batting_rates(
            batting_source,
            team_batting_source,
            2024,
            1,
        )
        assert len(result) == 1

        player = result[0]
        # vs-LHP rate with 0 PA regresses fully to league average
        league_hr_rate = 200 / 6000
        regression_pa = BATTING_SPLIT_REGRESSION_PA["hr"]
        expected_vs_lhp = weighted_rate(
            stats=[0.0],
            opportunities=[0.0],
            weights=[5],
            league_rate=league_hr_rate,
            regression_pa=regression_pa,
        )
        assert player.metadata["rates_vs_lhp"]["hr"] == pytest.approx(expected_vs_lhp, rel=1e-6)
        # Should be league average since 0 PA → pure regression
        assert player.metadata["rates_vs_lhp"]["hr"] == pytest.approx(league_hr_rate, rel=1e-6)

    def test_player_in_lhp_only(self, test_context: "Context") -> None:
        lhp_stats = _make_batting(player_id="p1", year=2023, pa=150, hr=5)
        league = _make_league_batting(year=2023)

        split_source = FakeSplitSource(
            vs_lhp={2023: [lhp_stats]},
            vs_rhp={2023: []},
        )
        delegate = FakePitchingDelegate()
        batting_source = _make_batting_source({})
        team_batting_source = _make_batting_source({2023: [league]})

        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
        )
        result = computer.compute_batting_rates(
            batting_source,
            team_batting_source,
            2024,
            1,
        )
        assert len(result) == 1

        player = result[0]
        league_hr_rate = 200 / 6000
        assert player.metadata["rates_vs_rhp"]["hr"] == pytest.approx(league_hr_rate, rel=1e-6)


class TestPlatoonPitchingDelegation:
    """Pitching delegates to wrapped computer."""

    def test_delegates_to_pitching_delegate(self, test_context: "Context") -> None:
        expected = [PlayerRates(player_id="sp1", name="Pitcher", year=2024, age=28, rates={"so": 0.25})]
        delegate = FakePitchingDelegate(pitching_result=expected)
        split_source = FakeSplitSource()
        pitching_source = _make_pitching_source({})
        team_pitching_source = _make_pitching_source({})

        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
        )
        result = computer.compute_pitching_rates(
            pitching_source,
            team_pitching_source,
            2024,
            3,
        )

        assert result == expected
        assert len(delegate.pitching_calls) == 1
        assert delegate.pitching_calls[0] == (pitching_source, team_pitching_source, 2024, 3)


class TestPlatoonPAPerYear:
    """pa_per_year sums both splits."""

    def test_pa_per_year_sums_splits(self, test_context: "Context") -> None:
        lhp = _make_batting(player_id="p1", year=2023, pa=150)
        rhp = _make_batting(player_id="p1", year=2023, pa=450)
        league = _make_league_batting(year=2023)

        split_source = FakeSplitSource(
            vs_lhp={2023: [lhp]},
            vs_rhp={2023: [rhp]},
        )
        delegate = FakePitchingDelegate()
        batting_source = _make_batting_source({})
        team_batting_source = _make_batting_source({2023: [league]})

        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
        )
        result = computer.compute_batting_rates(
            batting_source,
            team_batting_source,
            2024,
            1,
        )
        pa_per_year = result[0].metadata["pa_per_year"]
        assert pa_per_year == [600.0]

    def test_pa_per_year_multiyear(self, test_context: "Context") -> None:
        lhp_2023 = _make_batting(player_id="p1", year=2023, pa=150)
        rhp_2023 = _make_batting(player_id="p1", year=2023, pa=450)
        lhp_2022 = _make_batting(player_id="p1", year=2022, pa=100, age=27)
        rhp_2022 = _make_batting(player_id="p1", year=2022, pa=300, age=27)
        league = _make_league_batting()

        split_source = FakeSplitSource(
            vs_lhp={2023: [lhp_2023], 2022: [lhp_2022]},
            vs_rhp={2023: [rhp_2023], 2022: [rhp_2022]},
        )
        delegate = FakePitchingDelegate()
        batting_source = _make_batting_source({})
        team_batting_source = _make_batting_source(
            {2023: [league], 2022: [_make_league_batting(2022)]},
        )

        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
        )
        result = computer.compute_batting_rates(
            batting_source,
            team_batting_source,
            2024,
            2,
        )
        pa_per_year = result[0].metadata["pa_per_year"]
        assert pa_per_year == [600.0, 400.0]


class TestPlatoonHandCalculation:
    """Hand-calculated numerical verification with simple numbers."""

    def test_hand_calculated_blended_hr_rate(self, test_context: "Context") -> None:
        """Verify blended HR rate with known inputs.

        Setup:
        - 1 year back, weight = 5
        - vs LHP: 150 PA, 10 HR
        - vs RHP: 450 PA, 15 HR
        - League: 200 HR / 6000 PA = 0.0333...
        - Split regression PA for HR = 850 (425 * 2)
        - pct_vs_rhp = 0.72, pct_vs_lhp = 0.28
        """
        lhp = _make_batting(player_id="p1", year=2023, pa=150, hr=10)
        rhp = _make_batting(player_id="p1", year=2023, pa=450, hr=15)
        league = _make_league_batting(year=2023)

        split_source = FakeSplitSource(
            vs_lhp={2023: [lhp]},
            vs_rhp={2023: [rhp]},
        )
        delegate = FakePitchingDelegate()
        batting_source = _make_batting_source({})
        team_batting_source = _make_batting_source({2023: [league]})

        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
        )
        result = computer.compute_batting_rates(
            batting_source,
            team_batting_source,
            2024,
            1,
        )
        player = result[0]

        league_hr_rate = 200 / 6000  # 0.03333...

        # vs LHP: (5*10 + 850*0.03333) / (5*150 + 850)
        numerator_lhp = 5 * 10 + 850 * league_hr_rate
        denominator_lhp = 5 * 150 + 850
        expected_lhp = numerator_lhp / denominator_lhp

        # vs RHP: (5*15 + 850*0.03333) / (5*450 + 850)
        numerator_rhp = 5 * 15 + 850 * league_hr_rate
        denominator_rhp = 5 * 450 + 850
        expected_rhp = numerator_rhp / denominator_rhp

        expected_blended = 0.28 * expected_lhp + 0.72 * expected_rhp

        assert player.metadata["rates_vs_lhp"]["hr"] == pytest.approx(expected_lhp, rel=1e-6)
        assert player.metadata["rates_vs_rhp"]["hr"] == pytest.approx(expected_rhp, rel=1e-6)
        assert player.rates["hr"] == pytest.approx(expected_blended, rel=1e-6)


class TestPlatoonPlayerInfo:
    """Player info (id, name, year, age) populated correctly."""

    def test_player_metadata(self, test_context: "Context") -> None:
        lhp = _make_batting(player_id="p1", name="Mike Trout", year=2023, age=31)
        rhp = _make_batting(player_id="p1", name="Mike Trout", year=2023, age=31, team="LAA")
        league = _make_league_batting(year=2023)

        split_source = FakeSplitSource(
            vs_lhp={2023: [lhp]},
            vs_rhp={2023: [rhp]},
        )
        delegate = FakePitchingDelegate()
        batting_source = _make_batting_source({})
        team_batting_source = _make_batting_source({2023: [league]})

        computer = PlatoonRateComputer(
            split_source=split_source,
            pitching_delegate=delegate,
        )
        result = computer.compute_batting_rates(
            batting_source,
            team_batting_source,
            2024,
            1,
        )
        player = result[0]

        assert player.player_id == "p1"
        assert player.name == "Mike Trout"
        assert player.year == 2024
        assert player.age == 32  # 31 + (2024 - 2023)
