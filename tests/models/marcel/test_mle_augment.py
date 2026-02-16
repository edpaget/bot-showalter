import pytest

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.marcel.mle_augment import augment_inputs_with_mle
from fantasy_baseball_manager.models.marcel.types import MarcelInput, SeasonLine


BATTING_CATEGORIES = ("h", "doubles", "triples", "hr", "r", "rbi", "bb", "so", "sb", "cs", "hbp", "sf")

LEAGUE_RATES: dict[str, float] = {cat: 0.01 for cat in BATTING_CATEGORIES}
LEAGUE_RATES.update({"h": 0.045, "bb": 0.020, "so": 0.055, "hr": 0.008})

DISCOUNT_FACTOR = 0.55


def _make_mle_projection(
    player_id: int = 1,
    season: int = 2026,
    pa: int = 400,
    h: int = 100,
    doubles: int = 20,
    triples: int = 3,
    hr: int = 12,
    bb: int = 36,
    so: int = 80,
) -> Projection:
    return Projection(
        player_id=player_id,
        season=season,
        system="mle",
        version="v1",
        player_type="batter",
        stat_json={
            "pa": pa,
            "h": h,
            "doubles": doubles,
            "triples": triples,
            "hr": hr,
            "bb": bb,
            "so": so,
        },
    )


def _make_marcel_input(
    weighted_pt: float = 2000.0,
    age: int = 27,
) -> MarcelInput:
    rates = {cat: LEAGUE_RATES[cat] for cat in BATTING_CATEGORIES}
    rates["h"] = 0.050
    rates["hr"] = 0.010
    rates["bb"] = 0.022
    rates["so"] = 0.050
    return MarcelInput(
        weighted_rates=rates,
        weighted_pt=weighted_pt,
        league_rates=dict(LEAGUE_RATES),
        age=age,
        seasons=(SeasonLine(stats={cat: rates[cat] * 600 for cat in BATTING_CATEGORIES}, pa=600),),
    )


class TestAugmentInputsWithMle:
    def test_mle_only_player_creates_new_input(self) -> None:
        """No MLB data — creates a synthetic MarcelInput from MLE projection."""
        mle_proj = _make_mle_projection(player_id=99, pa=400)
        inputs: dict[int, MarcelInput] = {}

        result = augment_inputs_with_mle(
            inputs=inputs,
            mle_projections=[mle_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=DISCOUNT_FACTOR,
        )

        assert 99 in result
        mi = result[99]
        assert mi.weighted_pt == pytest.approx(400 * DISCOUNT_FACTOR)
        # Should have MLE rates for available categories
        assert mi.weighted_rates["h"] == pytest.approx(100 / 400)
        assert mi.weighted_rates["hr"] == pytest.approx(12 / 400)
        # Missing categories (r, rbi, sb, cs) should use league rates
        assert mi.weighted_rates["r"] == pytest.approx(LEAGUE_RATES["r"])
        assert mi.weighted_rates["rbi"] == pytest.approx(LEAGUE_RATES["rbi"])
        # Should have a SeasonLine with baseline PA
        assert len(mi.seasons) == 1
        assert mi.seasons[0].pa == 200  # Marcel default baseline

    def test_mle_only_player_uses_age_from_projection(self) -> None:
        """MLE-only player gets age from MLE projection stat_json."""
        mle_proj = _make_mle_projection(player_id=99, pa=400)
        # Add age to stat_json (as MLE model now provides)
        stat_json = dict(mle_proj.stat_json)
        stat_json["age"] = 22.5
        mle_proj = Projection(
            player_id=99,
            season=2026,
            system="mle",
            version="v1",
            player_type="batter",
            stat_json=stat_json,
        )

        result = augment_inputs_with_mle(
            inputs={},
            mle_projections=[mle_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=DISCOUNT_FACTOR,
        )

        assert result[99].age == 22

    def test_mixed_player_augments_existing(self) -> None:
        """Player has MLB data — MLE is PA-weight merged."""
        existing = _make_marcel_input(weighted_pt=1000.0)
        mle_proj = _make_mle_projection(player_id=1, pa=400)

        inputs = {1: existing}
        result = augment_inputs_with_mle(
            inputs=inputs,
            mle_projections=[mle_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=DISCOUNT_FACTOR,
        )

        mi = result[1]
        effective_mle_pa = 400 * DISCOUNT_FACTOR  # 220
        assert mi.weighted_pt == pytest.approx(1000.0 + effective_mle_pa)
        # Rates should be PA-weighted blend of existing and MLE
        # For 'h': existing=0.050 at 1000 PT, MLE=100/400=0.250 at 220 effective
        expected_h = (0.050 * 1000 + (100 / 400) * effective_mle_pa) / (1000 + effective_mle_pa)
        assert mi.weighted_rates["h"] == pytest.approx(expected_h, abs=1e-6)

    def test_established_player_mle_effect_minimal(self) -> None:
        """6000+ weighted PT — MLE negligible impact."""
        existing = _make_marcel_input(weighted_pt=6000.0)
        mle_proj = _make_mle_projection(player_id=1, pa=400, so=100)  # Moderate K%

        inputs = {1: existing}
        result = augment_inputs_with_mle(
            inputs=inputs,
            mle_projections=[mle_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=DISCOUNT_FACTOR,
        )

        mi = result[1]
        # 220 effective MLE PA vs 6000 MLB → ~3.5% MLE weight
        # so rate should barely change from existing 0.050
        effective_mle_pa = 400 * DISCOUNT_FACTOR
        mle_weight = effective_mle_pa / (6000.0 + effective_mle_pa)
        assert mle_weight < 0.04  # MLE contributes < 4%
        assert mi.weighted_rates["so"] == pytest.approx(existing.weighted_rates["so"], abs=0.010)

    def test_discount_factor_configurable(self) -> None:
        """Higher discount → more MLE weight."""
        mle_proj = _make_mle_projection(player_id=99, pa=400)

        result_low = augment_inputs_with_mle(
            inputs={},
            mle_projections=[mle_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=0.30,
        )
        result_high = augment_inputs_with_mle(
            inputs={},
            mle_projections=[mle_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=0.80,
        )

        assert result_low[99].weighted_pt < result_high[99].weighted_pt
        assert result_low[99].weighted_pt == pytest.approx(400 * 0.30)
        assert result_high[99].weighted_pt == pytest.approx(400 * 0.80)

    def test_mle_only_uses_league_rates(self) -> None:
        """MLE-only player gets provided league rates."""
        mle_proj = _make_mle_projection(player_id=99, pa=400)

        result = augment_inputs_with_mle(
            inputs={},
            mle_projections=[mle_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=DISCOUNT_FACTOR,
        )

        mi = result[99]
        assert mi.league_rates == LEAGUE_RATES

    def test_pitcher_mle_ignored(self) -> None:
        """Pitcher MLE projections skipped."""
        pitcher_proj = Projection(
            player_id=50,
            season=2026,
            system="mle",
            version="v1",
            player_type="pitcher",
            stat_json={"pa": 0, "ip": 100},
        )

        result = augment_inputs_with_mle(
            inputs={},
            mle_projections=[pitcher_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=DISCOUNT_FACTOR,
        )

        assert 50 not in result

    def test_zero_mle_pa_skipped(self) -> None:
        """MLE with 0 PA excluded."""
        mle_proj = _make_mle_projection(player_id=99, pa=0)

        result = augment_inputs_with_mle(
            inputs={},
            mle_projections=[mle_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=DISCOUNT_FACTOR,
        )

        assert 99 not in result

    def test_missing_categories_use_league_rate(self) -> None:
        """Categories not in MLE stat_json default to league rate."""
        mle_proj = _make_mle_projection(player_id=99, pa=400)

        result = augment_inputs_with_mle(
            inputs={},
            mle_projections=[mle_proj],
            categories=BATTING_CATEGORIES,
            league_rates=LEAGUE_RATES,
            discount_factor=DISCOUNT_FACTOR,
        )

        mi = result[99]
        # MLE stat_json has h, doubles, triples, hr, bb, so but NOT r, rbi, sb, cs, hbp, sf
        for cat in ("r", "rbi", "sb", "cs"):
            assert mi.weighted_rates[cat] == pytest.approx(LEAGUE_RATES[cat])
