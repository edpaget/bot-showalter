import pytest

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.marcel.statcast_augment import augment_inputs_with_statcast
from fantasy_baseball_manager.models.marcel.types import MarcelInput, SeasonLine


BATTING_CATEGORIES = ("h", "doubles", "triples", "hr", "r", "rbi", "bb", "so", "sb", "cs", "hbp", "sf")
PITCHING_CATEGORIES = ("w", "l", "sv", "h", "er", "hr", "bb", "so")

LEAGUE_BATTING_RATES: dict[str, float] = {cat: 0.01 for cat in BATTING_CATEGORIES}
LEAGUE_BATTING_RATES.update({"h": 0.045, "bb": 0.020, "so": 0.055, "hr": 0.008, "hbp": 0.003, "sf": 0.002})

LEAGUE_PITCHING_RATES: dict[str, float] = {cat: 0.01 for cat in PITCHING_CATEGORIES}
LEAGUE_PITCHING_RATES.update({"h": 0.80, "er": 0.40, "bb": 0.30, "so": 0.80, "hr": 0.10})


def _make_batter_input(
    player_id: int = 1,
    weighted_pt: float = 2000.0,
    age: int = 27,
    h_rate: float = 0.060,
    bb_rate: float = 0.022,
    hbp_rate: float = 0.003,
    sf_rate: float = 0.002,
) -> tuple[int, MarcelInput]:
    rates = {cat: LEAGUE_BATTING_RATES[cat] for cat in BATTING_CATEGORIES}
    rates["h"] = h_rate
    rates["bb"] = bb_rate
    rates["hbp"] = hbp_rate
    rates["sf"] = sf_rate
    return player_id, MarcelInput(
        weighted_rates=rates,
        weighted_pt=weighted_pt,
        league_rates=dict(LEAGUE_BATTING_RATES),
        age=age,
        seasons=(SeasonLine(stats={cat: rates[cat] * 600 for cat in BATTING_CATEGORIES}, pa=600),),
    )


def _make_pitcher_input(
    player_id: int = 10,
    weighted_pt: float = 500.0,
    age: int = 28,
    so_rate: float = 0.80,
    bb_rate: float = 0.30,
    hr_rate: float = 0.10,
    er_rate: float = 0.40,
    h_rate: float = 0.80,
) -> tuple[int, MarcelInput]:
    rates = {cat: LEAGUE_PITCHING_RATES[cat] for cat in PITCHING_CATEGORIES}
    rates["so"] = so_rate
    rates["bb"] = bb_rate
    rates["hr"] = hr_rate
    rates["er"] = er_rate
    rates["h"] = h_rate
    return player_id, MarcelInput(
        weighted_rates=rates,
        weighted_pt=weighted_pt,
        league_rates=dict(LEAGUE_PITCHING_RATES),
        age=age,
        seasons=(SeasonLine(stats={cat: rates[cat] * 180 for cat in PITCHING_CATEGORIES}, ip=180.0),),
    )


def _make_statcast_batter_projection(
    player_id: int = 1,
    season: int = 2024,
    avg: float = 0.280,
    obp: float = 0.350,
    slg: float = 0.450,
) -> Projection:
    return Projection(
        player_id=player_id,
        season=season,
        system="statcast-gbm",
        version="latest",
        player_type="batter",
        stat_json={"avg": avg, "obp": obp, "slg": slg, "woba": 0.350, "iso": slg - avg, "babip": 0.300},
    )


def _make_statcast_pitcher_projection(
    player_id: int = 10,
    season: int = 2024,
    k_per_9: float = 9.0,
    bb_per_9: float = 3.0,
    hr_per_9: float = 1.0,
    era: float = 3.60,
    whip: float = 1.20,
) -> Projection:
    return Projection(
        player_id=player_id,
        season=season,
        system="statcast-gbm",
        version="latest",
        player_type="pitcher",
        stat_json={
            "k_per_9": k_per_9,
            "bb_per_9": bb_per_9,
            "hr_per_9": hr_per_9,
            "era": era,
            "whip": whip,
            "fip": 3.50,
            "babip": 0.290,
        },
    )


class TestAugmentBatterRateBlending:
    def test_blends_h_rate_from_avg(self) -> None:
        """Statcast avg adjusts Marcel's h per-PA rate."""
        pid, mi = _make_batter_input(h_rate=0.060, bb_rate=0.022, hbp_rate=0.003, sf_rate=0.002)
        inputs = {pid: mi}
        # Marcel implied avg: h_rate / (1 - bb - hbp - sf) = 0.060 / 0.973 ≈ 0.06166
        # Statcast avg: 0.280
        # With weight=0.3: blended_avg = 0.06166*0.7 + 0.280*0.3 = 0.04316 + 0.084 = 0.12716
        # → new_h_rate = blended_avg * (1 - 0.022 - 0.003 - 0.002) = 0.12716 * 0.973
        proj = _make_statcast_batter_projection(player_id=pid, avg=0.280)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        assert pid in result
        new_h = result[pid].weighted_rates["h"]
        # h rate should move toward the statcast-implied h_per_pa
        assert new_h > mi.weighted_rates["h"]  # statcast avg (0.280) > marcel implied avg

    def test_blends_bb_rate_from_obp(self) -> None:
        """Statcast obp adjusts Marcel's bb per-PA rate (after h adjustment)."""
        pid, mi = _make_batter_input(h_rate=0.060, bb_rate=0.022, hbp_rate=0.003, sf_rate=0.002)
        inputs = {pid: mi}
        proj = _make_statcast_batter_projection(player_id=pid, avg=0.280, obp=0.380)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        new_bb = result[pid].weighted_rates["bb"]
        # statcast obp (0.380) >> marcel implied obp (~0.085) → bb should increase
        assert new_bb > mi.weighted_rates["bb"]

    def test_weighted_pt_unchanged(self) -> None:
        """Statcast blending doesn't add playing time weight."""
        pid, mi = _make_batter_input(weighted_pt=2000.0)
        inputs = {pid: mi}
        proj = _make_statcast_batter_projection(player_id=pid)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        assert result[pid].weighted_pt == mi.weighted_pt

    def test_weight_zero_no_change(self) -> None:
        """With blend_weight=0, rates are unchanged."""
        pid, mi = _make_batter_input()
        inputs = {pid: mi}
        proj = _make_statcast_batter_projection(player_id=pid)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.0,
        )

        for cat in BATTING_CATEGORIES:
            assert result[pid].weighted_rates[cat] == pytest.approx(mi.weighted_rates[cat], abs=1e-9)

    def test_weight_one_full_statcast(self) -> None:
        """With blend_weight=1.0, h rate is fully from statcast."""
        pid, mi = _make_batter_input(h_rate=0.060, bb_rate=0.022, hbp_rate=0.003, sf_rate=0.002)
        inputs = {pid: mi}
        statcast_avg = 0.280
        proj = _make_statcast_batter_projection(player_id=pid, avg=statcast_avg)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=1.0,
        )

        ab_fraction = 1 - 0.022 - 0.003 - 0.002
        expected_h_rate = statcast_avg * ab_fraction
        assert result[pid].weighted_rates["h"] == pytest.approx(expected_h_rate, abs=1e-6)


class TestAugmentPitcherRateBlending:
    def test_blends_so_rate_from_k_per_9(self) -> None:
        """Statcast k_per_9 adjusts Marcel's so per-IP rate."""
        pid, mi = _make_pitcher_input(so_rate=0.80)
        inputs = {pid: mi}
        proj = _make_statcast_pitcher_projection(player_id=pid, k_per_9=9.0)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        # statcast so_per_ip = 9.0 / 9 = 1.0
        # blended = 0.80 * 0.7 + 1.0 * 0.3 = 0.56 + 0.30 = 0.86
        assert result[pid].weighted_rates["so"] == pytest.approx(0.86, abs=1e-6)

    def test_blends_bb_rate_from_bb_per_9(self) -> None:
        """Statcast bb_per_9 adjusts Marcel's bb per-IP rate."""
        pid, mi = _make_pitcher_input(bb_rate=0.30)
        inputs = {pid: mi}
        proj = _make_statcast_pitcher_projection(player_id=pid, bb_per_9=2.7)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        # statcast bb_per_ip = 2.7 / 9 = 0.3, same as existing → no change
        assert result[pid].weighted_rates["bb"] == pytest.approx(0.30, abs=1e-6)

    def test_blends_hr_rate_from_hr_per_9(self) -> None:
        """Statcast hr_per_9 adjusts Marcel's hr per-IP rate."""
        pid, mi = _make_pitcher_input(hr_rate=0.10)
        inputs = {pid: mi}
        proj = _make_statcast_pitcher_projection(player_id=pid, hr_per_9=1.35)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        # statcast hr_per_ip = 1.35 / 9 = 0.15
        # blended = 0.10 * 0.7 + 0.15 * 0.3 = 0.07 + 0.045 = 0.115
        assert result[pid].weighted_rates["hr"] == pytest.approx(0.115, abs=1e-6)

    def test_blends_er_rate_from_era(self) -> None:
        """Statcast era adjusts Marcel's er per-IP rate."""
        pid, mi = _make_pitcher_input(er_rate=0.40)
        inputs = {pid: mi}
        proj = _make_statcast_pitcher_projection(player_id=pid, era=4.50)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        # statcast er_per_ip = 4.50 / 9 = 0.50
        # blended = 0.40 * 0.7 + 0.50 * 0.3 = 0.28 + 0.15 = 0.43
        assert result[pid].weighted_rates["er"] == pytest.approx(0.43, abs=1e-6)

    def test_blends_h_rate_from_whip(self) -> None:
        """Statcast whip adjusts Marcel's h per-IP rate (after bb adjustment)."""
        pid, mi = _make_pitcher_input(h_rate=0.80, bb_rate=0.30)
        inputs = {pid: mi}
        # whip = (h + bb) / ip → h_per_ip = whip - bb_per_ip
        proj = _make_statcast_pitcher_projection(player_id=pid, whip=1.10, bb_per_9=2.7)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        # statcast bb_per_ip = 2.7/9 = 0.30, blended bb = 0.30 (unchanged since same)
        # statcast h_per_ip = 1.10 - 0.30 = 0.80, same as existing → no change
        assert result[pid].weighted_rates["h"] == pytest.approx(0.80, abs=1e-6)

    def test_pitcher_weighted_pt_unchanged(self) -> None:
        """Statcast blending doesn't change pitcher weighted_pt."""
        pid, mi = _make_pitcher_input(weighted_pt=500.0)
        inputs = {pid: mi}
        proj = _make_statcast_pitcher_projection(player_id=pid)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        assert result[pid].weighted_pt == mi.weighted_pt


class TestAugmentEdgeCases:
    def test_no_matching_input_skipped(self) -> None:
        """Statcast projection for a player not in inputs is ignored."""
        pid, mi = _make_batter_input(player_id=1)
        inputs = {pid: mi}
        proj = _make_statcast_batter_projection(player_id=99)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        assert 99 not in result
        assert result[pid].weighted_rates["h"] == mi.weighted_rates["h"]

    def test_missing_statcast_rates_keeps_marcel_rates(self) -> None:
        """Categories not in statcast (r, rbi, sb, cs, etc.) stay unchanged."""
        pid, mi = _make_batter_input()
        inputs = {pid: mi}
        proj = _make_statcast_batter_projection(player_id=pid)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        for cat in ("r", "rbi", "sb", "cs", "doubles", "triples", "so"):
            assert result[pid].weighted_rates[cat] == mi.weighted_rates[cat]

    def test_preserves_age_and_seasons(self) -> None:
        """Augmentation only modifies weighted_rates, not age/seasons/league_rates."""
        pid, mi = _make_batter_input(age=30)
        inputs = {pid: mi}
        proj = _make_statcast_batter_projection(player_id=pid)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        assert result[pid].age == 30
        assert result[pid].seasons == mi.seasons
        assert result[pid].league_rates == mi.league_rates

    def test_batter_projection_not_applied_to_pitcher_input(self) -> None:
        """Batter statcast projection doesn't affect pitcher inputs."""
        pid, mi = _make_pitcher_input(player_id=10)
        inputs = {pid: mi}
        proj = _make_statcast_batter_projection(player_id=10)

        result = augment_inputs_with_statcast(
            inputs=inputs,
            statcast_projections=[proj],
            blend_weight=0.3,
        )

        # No batter-type matching for pitcher input — rates unchanged
        # (augmentation matches by player_type, not just player_id)
        for cat in PITCHING_CATEGORIES:
            assert result[pid].weighted_rates[cat] == mi.weighted_rates[cat]
