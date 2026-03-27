import pytest

from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.tier import PlayerTier
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.tier_generator import generate_tiers, tier_summary
from tests.fakes.repos import FakePlayerRepo


def _player(player_id: int, first: str, last: str) -> Player:
    return Player(name_first=first, name_last=last, id=player_id)


def _valuation(player_id: int, position: str, value: float, rank: int) -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2026,
        system="zar",
        version="1.0",
        projection_system="steamer",
        projection_version="2026",
        player_type=PlayerType.BATTER,
        position=position,
        value=value,
        rank=rank,
        category_scores={},
    )


class TestGapBasedTiers:
    def test_single_tier_no_gaps(self) -> None:
        """All close values → single tier."""
        players = [_player(1, "A", "One"), _player(2, "B", "Two"), _player(3, "C", "Three")]
        valuations = [
            _valuation(1, "1B", 30.0, 1),
            _valuation(2, "1B", 29.0, 2),
            _valuation(3, "1B", 28.0, 3),
        ]
        repo = FakePlayerRepo(players)
        result = generate_tiers(valuations, repo, method="gap", max_tiers=5)
        assert all(pt.tier == 1 for pt in result)

    def test_two_tiers_obvious_gap(self) -> None:
        """Obvious gap → two tiers."""
        players = [
            _player(1, "A", "One"),
            _player(2, "B", "Two"),
            _player(3, "C", "Three"),
            _player(4, "D", "Four"),
        ]
        valuations = [
            _valuation(1, "1B", 40.0, 1),
            _valuation(2, "1B", 38.0, 2),
            _valuation(3, "1B", 10.0, 3),
            _valuation(4, "1B", 8.0, 4),
        ]
        repo = FakePlayerRepo(players)
        result = generate_tiers(valuations, repo, method="gap", max_tiers=5)
        tier1 = [pt for pt in result if pt.tier == 1]
        tier2 = [pt for pt in result if pt.tier == 2]
        assert len(tier1) == 2
        assert len(tier2) == 2
        assert {pt.player_id for pt in tier1} == {1, 2}
        assert {pt.player_id for pt in tier2} == {3, 4}

    def test_multiple_positions_independent(self) -> None:
        """Each position is tiered independently."""
        players = [_player(i, "P", str(i)) for i in range(1, 9)]
        valuations = [
            # 1B: two tight clusters with a big gap between them
            _valuation(1, "1B", 40.0, 1),
            _valuation(2, "1B", 38.0, 2),
            _valuation(3, "1B", 10.0, 3),
            _valuation(4, "1B", 8.0, 4),
            # OF: all close values → single tier
            _valuation(5, "OF", 40.0, 1),
            _valuation(6, "OF", 39.0, 2),
            _valuation(7, "OF", 38.0, 3),
            _valuation(8, "OF", 37.0, 4),
        ]
        repo = FakePlayerRepo(players)
        result = generate_tiers(valuations, repo, method="gap", max_tiers=5)
        first_base = [pt for pt in result if pt.position == "1B"]
        outfield = [pt for pt in result if pt.position == "OF"]
        # 1B has a big gap → 2 tiers
        assert len({pt.tier for pt in first_base}) == 2
        # OF has close values → 1 tier
        assert len({pt.tier for pt in outfield}) == 1

    def test_equal_values_same_tier(self) -> None:
        """Players with equal values get the same tier."""
        players = [_player(1, "A", "One"), _player(2, "B", "Two"), _player(3, "C", "Three")]
        valuations = [
            _valuation(1, "1B", 30.0, 1),
            _valuation(2, "1B", 30.0, 2),
            _valuation(3, "1B", 30.0, 3),
        ]
        repo = FakePlayerRepo(players)
        result = generate_tiers(valuations, repo, method="gap", max_tiers=5)
        assert all(pt.tier == 1 for pt in result)

    def test_max_tiers_caps_output(self) -> None:
        """max_tiers limits the number of tiers produced."""
        players = [_player(i, "P", str(i)) for i in range(1, 8)]
        # Each player has a big gap to the next → would be 7 tiers without cap
        valuations = [_valuation(i, "1B", 100.0 - i * 10.0, i) for i in range(1, 8)]
        repo = FakePlayerRepo(players)
        result = generate_tiers(valuations, repo, method="gap", max_tiers=3)
        tiers = {pt.tier for pt in result}
        assert max(tiers) <= 3


class TestJenksTiers:
    def test_three_clear_clusters(self) -> None:
        """Three obvious clusters → three tiers."""
        players = [_player(i, "P", str(i)) for i in range(1, 10)]
        valuations = [
            # Cluster 1: ~50
            _valuation(1, "1B", 52.0, 1),
            _valuation(2, "1B", 50.0, 2),
            _valuation(3, "1B", 48.0, 3),
            # Cluster 2: ~30
            _valuation(4, "1B", 32.0, 4),
            _valuation(5, "1B", 30.0, 5),
            _valuation(6, "1B", 28.0, 6),
            # Cluster 3: ~10
            _valuation(7, "1B", 12.0, 7),
            _valuation(8, "1B", 10.0, 8),
            _valuation(9, "1B", 8.0, 9),
        ]
        repo = FakePlayerRepo(players)
        result = generate_tiers(valuations, repo, method="jenks", max_tiers=3)
        tiers = {pt.tier for pt in result}
        assert tiers == {1, 2, 3}
        # Top cluster is tier 1
        tier1_ids = {pt.player_id for pt in result if pt.tier == 1}
        assert tier1_ids == {1, 2, 3}

    def test_jenks_respects_max_tiers(self) -> None:
        """Jenks respects max_tiers parameter."""
        players = [_player(i, "P", str(i)) for i in range(1, 10)]
        valuations = [
            _valuation(1, "1B", 52.0, 1),
            _valuation(2, "1B", 50.0, 2),
            _valuation(3, "1B", 48.0, 3),
            _valuation(4, "1B", 32.0, 4),
            _valuation(5, "1B", 30.0, 5),
            _valuation(6, "1B", 28.0, 6),
            _valuation(7, "1B", 12.0, 7),
            _valuation(8, "1B", 10.0, 8),
            _valuation(9, "1B", 8.0, 9),
        ]
        repo = FakePlayerRepo(players)
        result = generate_tiers(valuations, repo, method="jenks", max_tiers=2)
        tiers = {pt.tier for pt in result}
        assert max(tiers) <= 2

    def test_jenks_equal_values_same_tier(self) -> None:
        """Equal values → same tier with Jenks."""
        players = [_player(1, "A", "One"), _player(2, "B", "Two"), _player(3, "C", "Three")]
        valuations = [
            _valuation(1, "1B", 30.0, 1),
            _valuation(2, "1B", 30.0, 2),
            _valuation(3, "1B", 30.0, 3),
        ]
        repo = FakePlayerRepo(players)
        result = generate_tiers(valuations, repo, method="jenks", max_tiers=3)
        assert all(pt.tier == 1 for pt in result)


class TestEdgeCases:
    def test_empty_valuations(self) -> None:
        """Empty input → empty output."""
        repo = FakePlayerRepo([])
        result = generate_tiers([], repo)
        assert result == []

    def test_single_player(self) -> None:
        """Single player → tier 1."""
        players = [_player(1, "A", "One")]
        valuations = [_valuation(1, "1B", 30.0, 1)]
        repo = FakePlayerRepo(players)
        result = generate_tiers(valuations, repo)
        assert len(result) == 1
        assert result[0].tier == 1
        assert result[0].rank == 1

    def test_unknown_method_raises(self) -> None:
        """Unknown clustering method → ValueError."""
        players = [_player(1, "A", "One")]
        valuations = [_valuation(1, "1B", 30.0, 1)]
        repo = FakePlayerRepo(players)
        with pytest.raises(ValueError, match="Unknown method"):
            generate_tiers(valuations, repo, method="kmeans")

    def test_missing_player_name_fallback(self) -> None:
        """Player not in repo → fallback name."""
        valuations = [_valuation(99, "1B", 30.0, 1)]
        repo = FakePlayerRepo([])  # no players
        result = generate_tiers(valuations, repo)
        assert len(result) == 1
        assert result[0].player_name == "Unknown (99)"


class TestTierSummary:
    def test_summary_counts_per_position_per_tier(self) -> None:
        """4 OF players split into 2 tiers → correct counts."""
        tiers = [
            PlayerTier(
                player_id=1,
                player_name="A One",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=1,
                value=40.0,
                rank=1,
            ),
            PlayerTier(
                player_id=2,
                player_name="B Two",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=1,
                value=38.0,
                rank=2,
            ),
            PlayerTier(
                player_id=3,
                player_name="C Three",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=2,
                value=10.0,
                rank=3,
            ),
            PlayerTier(
                player_id=4,
                player_name="D Four",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=2,
                value=8.0,
                rank=4,
            ),
        ]
        report = tier_summary(tiers)
        entries_by_tier = {e.tier: e for e in report.entries}
        assert entries_by_tier[1].count == 2
        assert entries_by_tier[2].count == 2

    def test_summary_total_and_avg_value(self) -> None:
        """Verify total_value and avg_value are correct for a known tier."""
        tiers = [
            PlayerTier(
                player_id=1,
                player_name="A One",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=1,
                value=40.0,
                rank=1,
            ),
            PlayerTier(
                player_id=2,
                player_name="B Two",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=1,
                value=30.0,
                rank=2,
            ),
        ]
        report = tier_summary(tiers)
        assert len(report.entries) == 1
        entry = report.entries[0]
        assert entry.total_value == 70.0
        assert entry.avg_value == 35.0

    def test_summary_best_player(self) -> None:
        """Best player is the highest-value player in that position+tier."""
        tiers = [
            PlayerTier(
                player_id=1,
                player_name="A One",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=1,
                value=40.0,
                rank=1,
            ),
            PlayerTier(
                player_id=2,
                player_name="B Two",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=1,
                value=30.0,
                rank=2,
            ),
        ]
        report = tier_summary(tiers)
        assert report.entries[0].best_player == "A One"

    def test_summary_empty_tiers(self) -> None:
        """Empty input → empty report."""
        report = tier_summary([])
        assert report.positions == []
        assert report.max_tier == 0
        assert report.entries == []

    def test_summary_multiple_positions(self) -> None:
        """OF and SP both appear in report, entries cover all combos."""
        tiers = [
            PlayerTier(
                player_id=1,
                player_name="A One",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=1,
                value=40.0,
                rank=1,
            ),
            PlayerTier(
                player_id=2,
                player_name="B Two",
                position="OF",
                player_type=PlayerType.BATTER,
                tier=2,
                value=20.0,
                rank=2,
            ),
            PlayerTier(
                player_id=3,
                player_name="C Three",
                position="SP",
                player_type=PlayerType.PITCHER,
                tier=1,
                value=35.0,
                rank=1,
            ),
        ]
        report = tier_summary(tiers)
        assert report.positions == ["OF", "SP"]
        assert report.max_tier == 2
        # 3 entries: (OF,1), (OF,2), (SP,1)
        assert len(report.entries) == 3
        keys = {(e.position, e.tier) for e in report.entries}
        assert keys == {("OF", 1), ("OF", 2), ("SP", 1)}
