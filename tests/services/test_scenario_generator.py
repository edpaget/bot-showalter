from __future__ import annotations

import dataclasses

import pytest

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.playing_time.engine import (
    ResidualBuckets,
    ResidualPercentiles,
)
from fantasy_baseball_manager.services.scenario_generator import (
    DEFAULT_SCENARIO_WEIGHTS,
    generate_pool_scenarios,
    generate_scenarios,
    scale_projection_to_pt,
)


def _make_batter_projection(
    player_id: int = 1,
    pa: float = 600.0,
    hr: float = 30.0,
    r: float = 90.0,
    rbi: float = 85.0,
    sb: float = 10.0,
    avg: float = 0.280,
    obp: float = 0.350,
) -> Projection:
    return Projection(
        player_id=player_id,
        season=2026,
        system="composite",
        version="v1",
        player_type="batter",
        stat_json={
            "pa": pa,
            "hr": hr,
            "r": r,
            "rbi": rbi,
            "sb": sb,
            "avg": avg,
            "obp": obp,
        },
    )


def _make_pitcher_projection(
    player_id: int = 2,
    ip: float = 180.0,
    w: float = 12.0,
    so: float = 200.0,
    era: float = 3.50,
    whip: float = 1.15,
) -> Projection:
    return Projection(
        player_id=player_id,
        season=2026,
        system="composite",
        version="v1",
        player_type="pitcher",
        stat_json={
            "ip": ip,
            "w": w,
            "so": so,
            "era": era,
            "whip": whip,
        },
    )


def _make_residual_percentiles(
    p10: float = -200.0,
    p25: float = -100.0,
    p50: float = 0.0,
    p75: float = 100.0,
    p90: float = 150.0,
) -> ResidualPercentiles:
    return ResidualPercentiles(p10=p10, p25=p25, p50=p50, p75=p75, p90=p90, count=100, std=120.0, mean_offset=0.0)


def _make_residual_buckets(player_type: str = "batter") -> ResidualBuckets:
    return ResidualBuckets(
        buckets={
            "young_healthy": _make_residual_percentiles(),
            "all": _make_residual_percentiles(),
        },
        player_type=player_type,
    )


class TestDefaultWeights:
    def test_sum_to_one(self) -> None:
        assert sum(DEFAULT_SCENARIO_WEIGHTS.values()) == pytest.approx(1.0)

    def test_has_five_percentiles(self) -> None:
        assert set(DEFAULT_SCENARIO_WEIGHTS.keys()) == {10, 25, 50, 75, 90}


class TestScaleProjectionToPt:
    def test_counting_stats_scale_proportionally(self) -> None:
        proj = _make_batter_projection(pa=600, hr=30, r=90)
        scaled = scale_projection_to_pt(proj, 300.0)
        assert scaled.stat_json["pa"] == pytest.approx(300.0)
        assert scaled.stat_json["hr"] == pytest.approx(15.0)
        assert scaled.stat_json["r"] == pytest.approx(45.0)

    def test_rate_stats_preserved(self) -> None:
        proj = _make_batter_projection(avg=0.280, obp=0.350)
        scaled = scale_projection_to_pt(proj, 300.0)
        assert scaled.stat_json["avg"] == pytest.approx(0.280)
        assert scaled.stat_json["obp"] == pytest.approx(0.350)

    def test_pitcher_rate_stats_preserved(self) -> None:
        proj = _make_pitcher_projection(era=3.50, whip=1.15)
        scaled = scale_projection_to_pt(proj, 90.0)
        assert scaled.stat_json["era"] == pytest.approx(3.50)
        assert scaled.stat_json["whip"] == pytest.approx(1.15)
        assert scaled.stat_json["so"] == pytest.approx(100.0)  # 200 * 90/180

    def test_batter_pa_clamped_to_750(self) -> None:
        proj = _make_batter_projection(pa=600)
        scaled = scale_projection_to_pt(proj, 900.0)
        assert scaled.stat_json["pa"] == pytest.approx(750.0)

    def test_batter_pa_clamped_to_zero(self) -> None:
        proj = _make_batter_projection(pa=600)
        scaled = scale_projection_to_pt(proj, -50.0)
        assert scaled.stat_json["pa"] == pytest.approx(0.0)

    def test_pitcher_ip_clamped_to_250(self) -> None:
        proj = _make_pitcher_projection(ip=180)
        scaled = scale_projection_to_pt(proj, 300.0)
        assert scaled.stat_json["ip"] == pytest.approx(250.0)

    def test_zero_pt_projection(self) -> None:
        proj = _make_batter_projection(pa=600, hr=30, r=90)
        scaled = scale_projection_to_pt(proj, 0.0)
        assert scaled.stat_json["pa"] == pytest.approx(0.0)
        assert scaled.stat_json["hr"] == pytest.approx(0.0)
        assert scaled.stat_json["r"] == pytest.approx(0.0)
        assert scaled.stat_json["avg"] == pytest.approx(0.280)

    def test_zero_original_pt(self) -> None:
        proj = _make_batter_projection(pa=0, hr=0, r=0)
        scaled = scale_projection_to_pt(proj, 300.0)
        # Can't scale from 0 — counting stats stay 0, PA becomes target
        assert scaled.stat_json["pa"] == pytest.approx(300.0)
        assert scaled.stat_json["hr"] == pytest.approx(0.0)

    def test_preserves_non_numeric_fields(self) -> None:
        proj = dataclasses.replace(
            _make_batter_projection(),
            stat_json={**_make_batter_projection().stat_json, "name": "Test Player"},
        )
        scaled = scale_projection_to_pt(proj, 300.0)
        assert scaled.stat_json["name"] == "Test Player"


class TestGenerateScenarios:
    def test_produces_correct_count(self) -> None:
        proj = _make_batter_projection(pa=600)
        percs = _make_residual_percentiles()
        scenarios = generate_scenarios(proj, percs, 600.0)
        assert len(scenarios) == 5

    def test_weights_sum_to_one(self) -> None:
        proj = _make_batter_projection(pa=600)
        percs = _make_residual_percentiles()
        scenarios = generate_scenarios(proj, percs, 600.0)
        total_weight = sum(w for _, w in scenarios)
        assert total_weight == pytest.approx(1.0)

    def test_scenario_pt_values(self) -> None:
        proj = _make_batter_projection(pa=600)
        percs = _make_residual_percentiles(p10=-200, p25=-100, p50=0, p75=100, p90=150)
        scenarios = generate_scenarios(proj, percs, 600.0)
        pts = [s.stat_json["pa"] for s, _ in scenarios]
        assert pts[0] == pytest.approx(400.0)  # 600 - 200
        assert pts[1] == pytest.approx(500.0)  # 600 - 100
        assert pts[2] == pytest.approx(600.0)  # 600 + 0
        assert pts[3] == pytest.approx(700.0)  # 600 + 100, clamped to 750? No, 700 < 750
        assert pts[4] == pytest.approx(750.0)  # 600 + 150 = 750

    def test_custom_weights(self) -> None:
        proj = _make_batter_projection(pa=600)
        percs = _make_residual_percentiles()
        custom = {10: 0.5, 90: 0.5}
        scenarios = generate_scenarios(proj, percs, 600.0, scenario_weights=custom)
        assert len(scenarios) == 2
        assert sum(w for _, w in scenarios) == pytest.approx(1.0)

    def test_counting_stats_scale_with_scenario(self) -> None:
        proj = _make_batter_projection(pa=600, hr=30)
        percs = _make_residual_percentiles(p50=0)
        scenarios = generate_scenarios(proj, percs, 600.0)
        # P50 scenario: PA stays 600, HR stays 30
        p50_proj, _ = scenarios[2]
        assert p50_proj.stat_json["hr"] == pytest.approx(30.0)


class TestGeneratePoolScenarios:
    def test_multiple_players(self) -> None:
        batter = _make_batter_projection(player_id=1, pa=600)
        pitcher = _make_pitcher_projection(player_id=2, ip=180)
        buckets_map = {
            "batter": _make_residual_buckets("batter"),
            "pitcher": _make_residual_buckets("pitcher"),
        }
        player_bucket_keys = {1: "young_healthy", 2: "young_healthy"}
        result = generate_pool_scenarios([batter, pitcher], buckets_map, player_bucket_keys)
        assert len(result) == 2
        assert len(result[1]) == 5
        assert len(result[2]) == 5

    def test_fallback_single_scenario(self) -> None:
        proj = _make_batter_projection(player_id=99, pa=500)
        buckets_map = {"batter": _make_residual_buckets("batter")}
        # Player 99 not in bucket keys → single scenario
        result = generate_pool_scenarios([proj], buckets_map, {})
        assert len(result[99]) == 1
        scenario_proj, weight = result[99][0]
        assert weight == pytest.approx(1.0)
        assert scenario_proj.stat_json["pa"] == pytest.approx(500.0)

    def test_missing_bucket_falls_back(self) -> None:
        proj = _make_batter_projection(player_id=1, pa=600)
        # Bucket key doesn't exist in buckets, but "all" does
        buckets_map = {"batter": _make_residual_buckets("batter")}
        player_bucket_keys = {1: "nonexistent_bucket"}
        result = generate_pool_scenarios([proj], buckets_map, player_bucket_keys)
        assert len(result[1]) == 5  # falls back to "all"

    def test_weights_sum_to_one_per_player(self) -> None:
        batter = _make_batter_projection(player_id=1, pa=600)
        buckets_map = {"batter": _make_residual_buckets("batter")}
        player_bucket_keys = {1: "young_healthy"}
        result = generate_pool_scenarios([batter], buckets_map, player_bucket_keys)
        total = sum(w for _, w in result[1])
        assert total == pytest.approx(1.0)
