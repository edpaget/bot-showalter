import pytest

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.pt_normalization import (
    build_consensus_lookup,
    normalize_projection_pt,
)


def _batter_projection(player_id: int = 1, **stat_overrides: object) -> Projection:
    stats: dict[str, object] = {
        "pa": 600,
        "hr": 30,
        "rbi": 100,
        "r": 90,
        "sb": 15,
        "avg": 0.280,
        "obp": 0.350,
        "slg": 0.500,
        "war": 5.0,
    }
    stats.update(stat_overrides)
    return Projection(
        player_id=player_id,
        season=2025,
        system="steamer",
        version="2025.1",
        player_type="batter",
        stat_json=stats,
    )


def _pitcher_projection(player_id: int = 1, **stat_overrides: object) -> Projection:
    stats: dict[str, object] = {
        "ip": 180,
        "so": 200,
        "w": 12,
        "bb": 50,
        "era": 3.20,
        "whip": 1.10,
        "war": 4.0,
    }
    stats.update(stat_overrides)
    return Projection(
        player_id=player_id,
        season=2025,
        system="steamer",
        version="2025.1",
        player_type="pitcher",
        stat_json=stats,
    )


class TestNormalizeProjectionPtBatter:
    def test_batter_counting_stats_rescaled(self) -> None:
        proj = _batter_projection(pa=600, hr=30, rbi=100, r=90, sb=15)
        result = normalize_projection_pt(proj, 500)
        assert result.stat_json["hr"] == pytest.approx(25.0)
        assert result.stat_json["rbi"] == pytest.approx(500 / 600 * 100)
        assert result.stat_json["r"] == pytest.approx(500 / 600 * 90)
        assert result.stat_json["sb"] == pytest.approx(500 / 600 * 15)

    def test_batter_rate_stats_unchanged(self) -> None:
        proj = _batter_projection(avg=0.280, obp=0.350, slg=0.500)
        result = normalize_projection_pt(proj, 500)
        assert result.stat_json["avg"] == 0.280
        assert result.stat_json["obp"] == 0.350
        assert result.stat_json["slg"] == 0.500

    def test_pa_replaced_with_consensus(self) -> None:
        proj = _batter_projection(pa=600)
        result = normalize_projection_pt(proj, 500)
        assert result.stat_json["pa"] == 500

    def test_missing_pt_returns_unmodified(self) -> None:
        stats = {"hr": 30, "avg": 0.280}
        proj = Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="2025.1",
            player_type="batter",
            stat_json=stats,
        )
        result = normalize_projection_pt(proj, 500)
        assert result.stat_json == stats

    def test_zero_pt_returns_unmodified(self) -> None:
        proj = _batter_projection(pa=0)
        result = normalize_projection_pt(proj, 500)
        assert result.stat_json["pa"] == 0

    def test_non_counting_non_rate_stat_unchanged(self) -> None:
        proj = _batter_projection(war=5.0)
        result = normalize_projection_pt(proj, 500)
        assert result.stat_json["war"] == 5.0


class TestNormalizeProjectionPtPitcher:
    def test_pitcher_counting_stats_rescaled(self) -> None:
        proj = _pitcher_projection(ip=180, so=200, w=12, bb=50)
        result = normalize_projection_pt(proj, 150)
        assert result.stat_json["so"] == pytest.approx(150 / 180 * 200)
        assert result.stat_json["w"] == pytest.approx(150 / 180 * 12)
        assert result.stat_json["bb"] == pytest.approx(150 / 180 * 50)

    def test_pitcher_rate_stats_unchanged(self) -> None:
        proj = _pitcher_projection(era=3.20, whip=1.10)
        result = normalize_projection_pt(proj, 150)
        assert result.stat_json["era"] == 3.20
        assert result.stat_json["whip"] == 1.10

    def test_ip_replaced_with_consensus(self) -> None:
        proj = _pitcher_projection(ip=180)
        result = normalize_projection_pt(proj, 150)
        assert result.stat_json["ip"] == 150


class TestBuildConsensusLookup:
    def test_averages_steamer_and_zips(self) -> None:
        steamer = [_batter_projection(player_id=1, pa=600)]
        zips = [_batter_projection(player_id=1, pa=500)]
        lookup = build_consensus_lookup(steamer, zips)
        assert lookup.batting_pt[1] == pytest.approx(550.0)

    def test_falls_back_to_single_system(self) -> None:
        steamer = [_batter_projection(player_id=1, pa=600)]
        zips: list[Projection] = []
        lookup = build_consensus_lookup(steamer, zips)
        assert lookup.batting_pt[1] == pytest.approx(600.0)

    def test_empty_when_no_projections(self) -> None:
        lookup = build_consensus_lookup([], [])
        assert lookup.batting_pt == {}
        assert lookup.pitching_pt == {}

    def test_separates_batters_and_pitchers(self) -> None:
        batter = _batter_projection(player_id=1, pa=600)
        pitcher = _pitcher_projection(player_id=2, ip=180)
        lookup = build_consensus_lookup([batter, pitcher], [])
        assert 1 in lookup.batting_pt
        assert 1 not in lookup.pitching_pt
        assert 2 in lookup.pitching_pt
        assert 2 not in lookup.batting_pt

    def test_averages_pitcher_ip(self) -> None:
        steamer = [_pitcher_projection(player_id=1, ip=180)]
        zips = [_pitcher_projection(player_id=1, ip=160)]
        lookup = build_consensus_lookup(steamer, zips)
        assert lookup.pitching_pt[1] == pytest.approx(170.0)
