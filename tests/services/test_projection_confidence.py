import sqlite3

import pytest

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.projection_confidence import (
    ConfidenceReport,
    PlayerConfidence,
    VarianceClassification,
)
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.projection_confidence import (
    classify_variance,
    compute_confidence,
)
from tests.helpers import seed_player


def _seed_projection(
    conn: sqlite3.Connection,
    player_id: int,
    system: str,
    stat_json: dict[str, float],
    *,
    season: int = 2026,
    version: str = "latest",
    player_type: str = "batter",
) -> None:
    repo = SqliteProjectionRepo(conn)
    repo.upsert(
        Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type=player_type,
            stat_json=stat_json,
        )
    )


def _test_league() -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=14,
        roster_pitchers=10,
        batting_categories=(
            CategoryConfig(key="hr", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="r", name="Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(
                key="obp",
                name="On-Base Percentage",
                stat_type=StatType.RATE,
                direction=Direction.HIGHER,
                numerator="h",
                denominator="ab",
            ),
        ),
        pitching_categories=(
            CategoryConfig(
                key="era",
                name="Earned Run Average",
                stat_type=StatType.RATE,
                direction=Direction.LOWER,
                numerator="er",
                denominator="ip",
            ),
            CategoryConfig(key="so", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
    )


def _get_projections(conn: sqlite3.Connection, season: int = 2026) -> list[Projection]:
    repo = SqliteProjectionRepo(conn)
    return repo.get_by_season(season)


class TestHighAgreement:
    def test_close_projections_yield_high_agreement(self, conn: sqlite3.Connection) -> None:
        """Three systems with close HR/R values should yield high agreement."""
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Juan Soto"},
        )

        assert len(report.players) == 1
        player = report.players[0]
        assert player.agreement_level == "high"
        assert player.overall_cv < 0.10


class TestLowAgreement:
    def test_divergent_projections_yield_low_agreement(self, conn: sqlite3.Connection) -> None:
        """Three systems with wildly different HR/R values should yield low agreement."""
        pid = seed_player(conn, name_first="Wild", name_last="Card", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 10, "r": 40, "obp": 0.300})
        _seed_projection(conn, pid, "zips", {"hr": 35, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "marcel", {"hr": 45, "r": 120, "obp": 0.320})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Wild Card"},
        )

        assert len(report.players) == 1
        player = report.players[0]
        assert player.agreement_level == "low"
        assert player.overall_cv >= 0.25


class TestMinSystemsFilter:
    def test_player_below_min_systems_excluded(self, conn: sqlite3.Connection) -> None:
        """Player with only 2 systems should be excluded when min_systems=3."""
        pid_few = seed_player(conn, name_first="Few", name_last="Systems", mlbam_id=100001)
        _seed_projection(conn, pid_few, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid_few, "zips", {"hr": 31, "r": 102, "obp": 0.390})

        pid_enough = seed_player(conn, name_first="Enough", name_last="Systems", mlbam_id=100002)
        _seed_projection(conn, pid_enough, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid_enough, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid_enough, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid_few: "Few Systems", pid_enough: "Enough Systems"},
            min_systems=3,
        )

        assert len(report.players) == 1
        assert report.players[0].player_name == "Enough Systems"


class TestRateStatExcludedFromOverallCV:
    def test_only_counting_stats_in_overall_cv(self, conn: sqlite3.Connection) -> None:
        """Counting stats at CV=0 + wildly divergent OBP => overall_cv == 0.0."""
        pid = seed_player(conn, name_first="Rate", name_last="Test", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.200})
        _seed_projection(conn, pid, "zips", {"hr": 30, "r": 100, "obp": 0.450})
        _seed_projection(conn, pid, "marcel", {"hr": 30, "r": 100, "obp": 0.350})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Rate Test"},
        )

        assert len(report.players) == 1
        assert report.players[0].overall_cv == 0.0


class TestSortOrder:
    def test_sorted_by_overall_cv_descending(self, conn: sqlite3.Connection) -> None:
        """Report should be sorted by overall_cv descending (most uncertain first)."""
        pid_safe = seed_player(conn, name_first="Safe", name_last="Player", mlbam_id=100001)
        _seed_projection(conn, pid_safe, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid_safe, "zips", {"hr": 30, "r": 100, "obp": 0.390})
        _seed_projection(conn, pid_safe, "marcel", {"hr": 30, "r": 100, "obp": 0.410})

        pid_risky = seed_player(conn, name_first="Risky", name_last="Player", mlbam_id=100002)
        _seed_projection(conn, pid_risky, "steamer", {"hr": 10, "r": 40, "obp": 0.300})
        _seed_projection(conn, pid_risky, "zips", {"hr": 35, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid_risky, "marcel", {"hr": 45, "r": 120, "obp": 0.320})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid_safe: "Safe Player", pid_risky: "Risky Player"},
        )

        assert len(report.players) == 2
        assert report.players[0].player_name == "Risky Player"
        assert report.players[1].player_name == "Safe Player"
        assert report.players[0].overall_cv >= report.players[1].overall_cv


class TestVersionDedup:
    def test_same_system_different_versions_deduped(self, conn: sqlite3.Connection) -> None:
        """Two versions of the same system should count as one system, not two."""
        pid = seed_player(conn, name_first="Dedup", name_last="Test", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.400}, version="v1")
        _seed_projection(conn, pid, "steamer", {"hr": 32, "r": 105, "obp": 0.410}, version="v2")
        _seed_projection(conn, pid, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.380})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Dedup Test"},
            min_systems=3,
        )

        assert len(report.players) == 1
        # Should have 3 systems (steamer, zips, marcel), not 4
        hr_spread = next(s for s in report.players[0].spreads if s.stat == "hr")
        assert len(hr_spread.systems) == 3


class TestPositions:
    def test_position_from_positions_dict(self, conn: sqlite3.Connection) -> None:
        """Position should come from the positions dict if provided."""
        pid = seed_player(conn, name_first="Pos", name_last="Test", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Pos Test"},
            positions={pid: "SS"},
        )

        assert report.players[0].position == "SS"

    def test_position_defaults_to_empty(self, conn: sqlite3.Connection) -> None:
        """Position should default to empty string when no positions dict."""
        pid = seed_player(conn, name_first="No", name_last="Pos", mlbam_id=100001)
        _seed_projection(conn, pid, "steamer", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "zips", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "No Pos"},
        )

        assert report.players[0].position == ""


class TestPitcherStats:
    def test_pitcher_uses_pitching_categories(self, conn: sqlite3.Connection) -> None:
        """Pitchers should use pitching categories (ERA, SO), not batting."""
        pid = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        _seed_projection(conn, pid, "steamer", {"era": 3.00, "so": 200, "ip": 180, "er": 60}, player_type="pitcher")
        _seed_projection(conn, pid, "zips", {"era": 3.20, "so": 190, "ip": 175, "er": 62}, player_type="pitcher")
        _seed_projection(conn, pid, "marcel", {"era": 3.50, "so": 180, "ip": 170, "er": 66}, player_type="pitcher")
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Gerrit Cole"},
        )

        assert len(report.players) == 1
        stat_names = {s.stat for s in report.players[0].spreads}
        assert "so" in stat_names
        assert "era" in stat_names
        assert "hr" not in stat_names
        assert "r" not in stat_names


class TestSystemsListInReport:
    def test_systems_sorted_and_deduped(self, conn: sqlite3.Connection) -> None:
        """report.systems should be sorted and deduplicated."""
        pid = seed_player(conn, name_first="Test", name_last="Player", mlbam_id=100001)
        _seed_projection(conn, pid, "zips", {"hr": 30, "r": 100, "obp": 0.400})
        _seed_projection(conn, pid, "steamer", {"hr": 31, "r": 102, "obp": 0.390})
        _seed_projection(conn, pid, "marcel", {"hr": 29, "r": 98, "obp": 0.410})
        # Second version of steamer
        _seed_projection(conn, pid, "steamer", {"hr": 32, "r": 105, "obp": 0.420}, version="v2")
        conn.commit()

        report = compute_confidence(
            _get_projections(conn),
            _test_league(),
            {pid: "Test Player"},
        )

        assert report.systems == ["marcel", "steamer", "zips"]


class TestEmptyInputs:
    def test_no_projections_yields_empty_report(self, conn: sqlite3.Connection) -> None:
        """No projections should yield an empty report."""
        report = compute_confidence(
            [],
            _test_league(),
            {},
        )

        assert report.players == []
        assert report.systems == []


# ---------------------------------------------------------------------------
# Helpers for classify_variance tests (pure — no DB needed)
# ---------------------------------------------------------------------------


def _player_confidence(
    player_id: int = 1,
    player_name: str = "Test Player",
    agreement_level: str = "medium",
    overall_cv: float = 0.15,
) -> PlayerConfidence:
    return PlayerConfidence(
        player_id=player_id,
        player_name=player_name,
        player_type="batter",
        position="OF",
        spreads=[],
        overall_cv=overall_cv,
        agreement_level=agreement_level,
    )


def _cv_report(*players: PlayerConfidence) -> ConfidenceReport:
    return ConfidenceReport(season=2026, systems=["steamer", "zips", "marcel"], players=list(players))


def _cv_valuation(
    player_id: int = 1,
    value: float = 20.0,
    projection_system: str = "steamer",
) -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2026,
        system="default",
        version="latest",
        projection_system=projection_system,
        projection_version="latest",
        player_type="batter",
        position="OF",
        value=value,
        rank=1,
        category_scores={},
    )


def _cv_adp(player_id: int = 1, overall_pick: float = 10.0) -> ADP:
    return ADP(
        player_id=player_id,
        season=2026,
        provider="nfbc",
        overall_pick=overall_pick,
        rank=1,
        positions="OF",
    )


def _player_vals(player_id: int, values: list[float]) -> list[Valuation]:
    """Create valuations from projection systems for one player."""
    systems = ["steamer", "zips", "marcel", "statcast"]
    return [
        _cv_valuation(player_id=player_id, value=v, projection_system=s) for v, s in zip(values, systems, strict=False)
    ]


def _background_vals(count: int = 25, start_value: float = 50.0) -> list[Valuation]:
    """Create background players (IDs 900+) with descending median values."""
    vals: list[Valuation] = []
    for i in range(count):
        v = start_value - i
        vals.extend(_player_vals(900 + i, [v, v, v]))
    return vals


# ---------------------------------------------------------------------------
# classify_variance tests
# ---------------------------------------------------------------------------


class TestSafeConsensus:
    def test_high_agreement_adp_close_to_value_rank(self) -> None:
        """High agreement + ADP near value rank → SAFE_CONSENSUS."""
        pc = _player_confidence(player_id=1, agreement_level="high", overall_cv=0.05)
        report = _cv_report(pc)
        vals = _player_vals(1, [28.0, 30.0, 32.0])
        adps = [_cv_adp(player_id=1, overall_pick=1.0)]

        result = classify_variance(report, vals, adps)

        assert len(result) == 1
        assert result[0].classification == VarianceClassification.SAFE_CONSENSUS

    def test_safe_consensus_at_threshold_boundary(self) -> None:
        """ADP exactly 10 ranks from value rank still counts as close."""
        pc = _player_confidence(player_id=1, agreement_level="high", overall_cv=0.05)
        report = _cv_report(pc)
        vals = _player_vals(1, [28.0, 30.0, 32.0])
        bg = _background_vals(count=15, start_value=25.0)
        # value_rank=1 (median=30 tops bg's 25), adp_rank=11 → |rank_diff|=10
        adps = [_cv_adp(player_id=1, overall_pick=11.0)]

        result = classify_variance(report, vals + bg, adps)

        assert result[0].classification == VarianceClassification.SAFE_CONSENSUS


class TestKnownQuantity:
    def test_high_agreement_adp_far_from_value_rank(self) -> None:
        """High agreement + ADP far from value rank → KNOWN_QUANTITY."""
        pc = _player_confidence(player_id=1, agreement_level="high", overall_cv=0.05)
        report = _cv_report(pc)
        vals = _player_vals(1, [28.0, 30.0, 32.0])
        bg = _background_vals(count=20, start_value=25.0)
        # value_rank=1, adp_rank=15 → |rank_diff|=14 > 10
        adps = [_cv_adp(player_id=1, overall_pick=15.0)]

        result = classify_variance(report, vals + bg, adps)

        assert result[0].classification == VarianceClassification.KNOWN_QUANTITY

    def test_high_agreement_no_adp(self) -> None:
        """High agreement + no ADP → KNOWN_QUANTITY."""
        pc = _player_confidence(player_id=1, agreement_level="high", overall_cv=0.05)
        report = _cv_report(pc)
        vals = _player_vals(1, [28.0, 30.0, 32.0])

        result = classify_variance(report, vals)

        assert result[0].classification == VarianceClassification.KNOWN_QUANTITY

    def test_medium_agreement_no_adp(self) -> None:
        """Medium agreement + no ADP → KNOWN_QUANTITY."""
        pc = _player_confidence(player_id=1, agreement_level="medium", overall_cv=0.15)
        report = _cv_report(pc)
        vals = _player_vals(1, [20.0, 25.0, 30.0])

        result = classify_variance(report, vals)

        assert result[0].classification == VarianceClassification.KNOWN_QUANTITY

    def test_medium_agreement_no_upside(self) -> None:
        """Medium agreement + ADP + no significant upside → KNOWN_QUANTITY."""
        pc = _player_confidence(player_id=1, agreement_level="medium", overall_cv=0.15)
        report = _cv_report(pc)
        vals = _player_vals(1, [19.0, 20.0, 21.0])
        adps = [_cv_adp(player_id=1, overall_pick=1.0)]

        result = classify_variance(report, vals, adps)

        assert result[0].classification == VarianceClassification.KNOWN_QUANTITY


class TestUpsideGamble:
    def test_low_agreement_adp_late_with_upside(self) -> None:
        """Low agreement + ADP late + upside → UPSIDE_GAMBLE."""
        pc = _player_confidence(player_id=1, agreement_level="low", overall_cv=0.30)
        report = _cv_report(pc)
        vals = _player_vals(1, [5.0, 15.0, 40.0])
        bg = _background_vals(count=30, start_value=50.0)
        # value_rank=1 (median=15 < bg start 50, actually no...)
        # bg values: 50,49,...,21. Player median=15 → value_rank=31
        # Hmm, that pushes player to rank 31, not 1
        # ADP pick 55 → rank_diff=55-31=24 > 20 → late
        adps = [_cv_adp(player_id=1, overall_pick=55.0)]

        result = classify_variance(report, vals + bg, adps)

        assert result[0].classification == VarianceClassification.UPSIDE_GAMBLE
        assert result[0].risk_reward_score > 0

    def test_low_agreement_neutral_adp_with_upside(self) -> None:
        """Low agreement + neutral ADP + upside → UPSIDE_GAMBLE."""
        pc = _player_confidence(player_id=1, agreement_level="low", overall_cv=0.30)
        report = _cv_report(pc)
        vals = _player_vals(1, [5.0, 15.0, 40.0])
        # ADP=1, value_rank=1 → rank_diff=0, neither late nor early
        adps = [_cv_adp(player_id=1, overall_pick=1.0)]

        result = classify_variance(report, vals, adps)

        assert result[0].classification == VarianceClassification.UPSIDE_GAMBLE


class TestRiskyAvoid:
    def test_low_agreement_adp_early(self) -> None:
        """Low agreement + ADP early → RISKY_AVOID."""
        pc = _player_confidence(player_id=1, agreement_level="low", overall_cv=0.30)
        report = _cv_report(pc)
        vals = _player_vals(1, [8.0, 10.0, 12.0])
        bg = _background_vals(count=25, start_value=50.0)
        # bg values: 50,49,...,26. Player median=10 → value_rank=26
        # ADP pick 3 → rank_diff=3-26=-23 < -20 → early
        adps = [_cv_adp(player_id=1, overall_pick=3.0)]

        result = classify_variance(report, vals + bg, adps)

        assert result[0].classification == VarianceClassification.RISKY_AVOID
        assert result[0].risk_reward_score < 0

    def test_low_agreement_no_upside(self) -> None:
        """Low agreement + no significant upside → RISKY_AVOID."""
        pc = _player_confidence(player_id=1, agreement_level="low", overall_cv=0.30)
        report = _cv_report(pc)
        vals = _player_vals(1, [9.0, 10.0, 11.0])
        adps = [_cv_adp(player_id=1, overall_pick=1.0)]

        result = classify_variance(report, vals, adps)

        assert result[0].classification == VarianceClassification.RISKY_AVOID


class TestHiddenUpside:
    def test_medium_agreement_significant_upside(self) -> None:
        """Medium agreement + significant upside → HIDDEN_UPSIDE."""
        pc = _player_confidence(player_id=1, agreement_level="medium", overall_cv=0.15)
        report = _cv_report(pc)
        # max(35) >= 1.3 * adp_expected(20) = 26
        vals = _player_vals(1, [10.0, 20.0, 35.0])
        adps = [_cv_adp(player_id=1, overall_pick=1.0)]

        result = classify_variance(report, vals, adps)

        assert result[0].classification == VarianceClassification.HIDDEN_UPSIDE

    def test_low_agreement_no_adp(self) -> None:
        """Low agreement + no ADP → HIDDEN_UPSIDE."""
        pc = _player_confidence(player_id=1, agreement_level="low", overall_cv=0.30)
        report = _cv_report(pc)
        vals = _player_vals(1, [5.0, 15.0, 40.0])

        result = classify_variance(report, vals)

        assert result[0].classification == VarianceClassification.HIDDEN_UPSIDE


class TestRiskRewardScore:
    def test_positive_for_upside_play(self) -> None:
        """Risk-reward score positive when upside exceeds downside."""
        pc = _player_confidence(player_id=1, agreement_level="low", overall_cv=0.30)
        report = _cv_report(pc)
        # max=40, min=5, median=15 → adp_expected=15
        # rr = 40 + 5 - 2*15 = 15
        vals = _player_vals(1, [5.0, 15.0, 40.0])
        adps = [_cv_adp(player_id=1, overall_pick=1.0)]

        result = classify_variance(report, vals, adps)

        assert result[0].risk_reward_score == pytest.approx(15.0)

    def test_negative_for_risky_play(self) -> None:
        """Risk-reward score negative when downside exceeds upside."""
        pc = _player_confidence(player_id=1, agreement_level="low", overall_cv=0.30)
        report = _cv_report(pc)
        vals = _player_vals(1, [8.0, 10.0, 12.0])
        bg = _background_vals(count=25, start_value=50.0)
        # value_rank=26, adp_rank=3 → adp_expected=rank_to_value[3]=48
        # rr = 12 + 8 - 2*48 = -76
        adps = [_cv_adp(player_id=1, overall_pick=3.0)]

        result = classify_variance(report, vals + bg, adps)

        assert result[0].risk_reward_score == pytest.approx(-76.0)

    def test_zero_for_symmetric_spread(self) -> None:
        """Symmetric spread around adp_expected → risk_reward_score ≈ 0."""
        pc = _player_confidence(player_id=1, agreement_level="medium", overall_cv=0.15)
        report = _cv_report(pc)
        # max=30, min=10, median=20 → adp_expected=20
        # rr = 30 + 10 - 2*20 = 0
        vals = _player_vals(1, [10.0, 20.0, 30.0])
        adps = [_cv_adp(player_id=1, overall_pick=1.0)]

        result = classify_variance(report, vals, adps)

        assert result[0].risk_reward_score == pytest.approx(0.0)


class TestValueRank:
    def test_value_rank_by_median_descending(self) -> None:
        """Value rank 1 goes to highest median value, 2 to next, etc."""
        pc1 = _player_confidence(player_id=1, player_name="Top", agreement_level="high", overall_cv=0.05)
        pc2 = _player_confidence(player_id=2, player_name="Mid", agreement_level="high", overall_cv=0.05)
        pc3 = _player_confidence(player_id=3, player_name="Low", agreement_level="high", overall_cv=0.05)
        report = _cv_report(pc1, pc2, pc3)
        vals = (
            _player_vals(1, [28.0, 30.0, 32.0])  # median=30
            + _player_vals(2, [18.0, 20.0, 22.0])  # median=20
            + _player_vals(3, [8.0, 10.0, 12.0])  # median=10
        )

        result = classify_variance(report, vals)

        by_name = {r.player.player_name: r for r in result}
        assert by_name["Top"].value_rank == 1
        assert by_name["Mid"].value_rank == 2
        assert by_name["Low"].value_rank == 3


class TestClassifyEdgeCases:
    def test_player_without_valuations_excluded(self) -> None:
        """Player in report but not in valuations → excluded."""
        pc = _player_confidence(player_id=1, agreement_level="high")
        report = _cv_report(pc)

        result = classify_variance(report, [])

        assert result == []

    def test_empty_report(self) -> None:
        """Empty report → empty result."""
        report = _cv_report()

        result = classify_variance(report, [])

        assert result == []

    def test_adp_rank_beyond_pool_still_classifies(self) -> None:
        """ADP rank beyond pool size → still classifies with clamped value."""
        pc = _player_confidence(player_id=1, agreement_level="high", overall_cv=0.05)
        report = _cv_report(pc)
        vals = _player_vals(1, [28.0, 30.0, 32.0])
        adps = [_cv_adp(player_id=1, overall_pick=100.0)]

        result = classify_variance(report, vals, adps)

        assert len(result) == 1
        assert result[0].adp_rank == 100
        # rank_diff=100-1=99, |99|>10, high → KNOWN_QUANTITY
        assert result[0].classification == VarianceClassification.KNOWN_QUANTITY

    def test_multiple_adp_entries_takes_lowest_pick(self) -> None:
        """Multiple ADP entries per player → use lowest overall_pick."""
        pc = _player_confidence(player_id=1, agreement_level="high", overall_cv=0.05)
        report = _cv_report(pc)
        vals = _player_vals(1, [28.0, 30.0, 32.0])
        adps = [
            _cv_adp(player_id=1, overall_pick=15.0),
            _cv_adp(player_id=1, overall_pick=5.0),
            _cv_adp(player_id=1, overall_pick=10.0),
        ]

        result = classify_variance(report, vals, adps)

        assert result[0].adp_rank == 5
