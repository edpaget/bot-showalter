import pytest

from fantasy_baseball_manager.marcel.league_averages import (
    compute_batting_league_rates,
    compute_pitching_league_rates,
    rebaseline,
)
from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)


def _make_batting_stats(
    *,
    pa: int = 1000,
    ab: int = 900,
    h: int = 250,
    singles: int = 160,
    doubles: int = 50,
    triples: int = 5,
    hr: int = 35,
    bb: int = 80,
    so: int = 200,
    hbp: int = 10,
    sf: int = 5,
    sh: int = 5,
    sb: int = 20,
    cs: int = 5,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id="league",
        name="League",
        year=2024,
        age=27,
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
    )


class TestComputeBattingLeagueRates:
    def test_single_team(self) -> None:
        stats = [_make_batting_stats(pa=1000, hr=35, bb=80, so=200)]
        rates = compute_batting_league_rates(stats)
        assert rates["hr"] == pytest.approx(35 / 1000)
        assert rates["bb"] == pytest.approx(80 / 1000)
        assert rates["so"] == pytest.approx(200 / 1000)

    def test_multiple_teams_aggregated(self) -> None:
        team_a = _make_batting_stats(
            pa=500,
            hr=20,
            bb=40,
            so=100,
            doubles=25,
            triples=2,
            singles=80,
            h=127,
            hbp=5,
            sf=3,
            sh=2,
            sb=10,
            cs=3,
            ab=450,
        )
        team_b = _make_batting_stats(
            pa=500,
            hr=15,
            bb=50,
            so=120,
            doubles=30,
            triples=3,
            singles=70,
            h=118,
            hbp=4,
            sf=2,
            sh=4,
            sb=8,
            cs=4,
            ab=440,
        )
        rates = compute_batting_league_rates([team_a, team_b])
        assert rates["hr"] == pytest.approx(35 / 1000)
        assert rates["bb"] == pytest.approx(90 / 1000)
        assert rates["so"] == pytest.approx(220 / 1000)

    def test_all_component_stats_present(self) -> None:
        stats = [_make_batting_stats()]
        rates = compute_batting_league_rates(stats)
        expected_keys = {"singles", "doubles", "triples", "hr", "bb", "so", "hbp", "sf", "sh", "sb", "cs"}
        assert set(rates.keys()) == expected_keys


class TestComputePitchingLeagueRates:
    def test_single_team(self) -> None:
        stats = [
            PitchingSeasonStats(
                player_id="league",
                name="League",
                year=2024,
                age=27,
                ip=1450.0,
                g=162,
                gs=162,
                er=650,
                h=1350,
                bb=500,
                so=1400,
                hr=180,
                hbp=60,
            )
        ]
        outs = 1450.0 * 3
        rates = compute_pitching_league_rates(stats)
        assert rates["so"] == pytest.approx(1400 / outs)
        assert rates["hr"] == pytest.approx(180 / outs)
        assert rates["bb"] == pytest.approx(500 / outs)


class TestRebaseline:
    def test_identity_when_source_equals_target(self) -> None:
        # If source and target league rates are the same, player rate unchanged
        projected = {"hr": 0.05, "bb": 0.10}
        source = {"hr": 0.035, "bb": 0.08}
        target = {"hr": 0.035, "bb": 0.08}
        result = rebaseline(projected, source, target)
        assert result["hr"] == pytest.approx(0.05)
        assert result["bb"] == pytest.approx(0.10)

    def test_scales_by_league_ratio(self) -> None:
        # If target league rate is double the source, player rate doubles
        projected = {"hr": 0.05, "bb": 0.10}
        source = {"hr": 0.02, "bb": 0.05}
        target = {"hr": 0.04, "bb": 0.10}
        result = rebaseline(projected, source, target)
        assert result["hr"] == pytest.approx(0.10)
        assert result["bb"] == pytest.approx(0.20)

    def test_partial_scaling(self) -> None:
        # Target HR rate is 1.5x source, player rate scales by 1.5x
        projected = {"hr": 0.06}
        source = {"hr": 0.03}
        target = {"hr": 0.045}
        result = rebaseline(projected, source, target)
        assert result["hr"] == pytest.approx(0.09)

    def test_zero_source_rate_preserves_player_rate(self) -> None:
        projected = {"hr": 0.05}
        source = {"hr": 0.0}
        target = {"hr": 0.04}
        result = rebaseline(projected, source, target)
        assert result["hr"] == pytest.approx(0.05)
