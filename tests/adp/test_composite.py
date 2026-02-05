"""Tests for CompositeADPSource."""

from datetime import UTC, datetime

from fantasy_baseball_manager.adp.composite import CompositeADPSource
from fantasy_baseball_manager.adp.models import ADPData, ADPEntry


class MockADPSource:
    """Mock ADP source for testing."""

    def __init__(self, entries: tuple[ADPEntry, ...], source: str) -> None:
        self._entries = entries
        self._source = source

    def fetch_adp(self) -> ADPData:
        return ADPData(
            entries=self._entries,
            fetched_at=datetime.now(UTC),
            source=self._source,
        )


class TestCompositeADPSource:
    """Tests for CompositeADPSource."""

    def test_single_source_passthrough(self) -> None:
        """Test that single source entries pass through unchanged."""
        entries = (
            ADPEntry(name="Mike Trout", adp=1.5, positions=("OF",)),
            ADPEntry(name="Shohei Ohtani", adp=2.3, positions=("DH", "SP")),
        )
        source = MockADPSource(entries, "mock")
        composite = CompositeADPSource([source])

        result = composite.fetch_adp()

        assert result.source == "composite"
        assert len(result.entries) == 2
        assert result.entries[0].name == "Mike Trout"
        assert result.entries[0].adp == 1.5

    def test_averages_adp_across_sources(self) -> None:
        """Test that ADP is averaged across multiple sources."""
        source1 = MockADPSource(
            (ADPEntry(name="Mike Trout", adp=1.0, positions=("OF",)),),
            "source1",
        )
        source2 = MockADPSource(
            (ADPEntry(name="Mike Trout", adp=3.0, positions=("OF",)),),
            "source2",
        )
        composite = CompositeADPSource([source1, source2])

        result = composite.fetch_adp()

        assert len(result.entries) == 1
        assert result.entries[0].name == "Mike Trout"
        assert result.entries[0].adp == 2.0  # Average of 1.0 and 3.0

    def test_handles_players_in_only_one_source(self) -> None:
        """Test players appearing in only one source are included."""
        source1 = MockADPSource(
            (
                ADPEntry(name="Mike Trout", adp=1.0, positions=("OF",)),
                ADPEntry(name="Only Source 1", adp=50.0, positions=("1B",)),
            ),
            "source1",
        )
        source2 = MockADPSource(
            (
                ADPEntry(name="Mike Trout", adp=2.0, positions=("OF",)),
                ADPEntry(name="Only Source 2", adp=60.0, positions=("SS",)),
            ),
            "source2",
        )
        composite = CompositeADPSource([source1, source2])

        result = composite.fetch_adp()

        # Should have 3 unique players
        names = {e.name for e in result.entries}
        assert names == {"Mike Trout", "Only Source 1", "Only Source 2"}

    def test_normalizes_names_for_matching(self) -> None:
        """Test that names are normalized for matching across sources."""
        source1 = MockADPSource(
            (ADPEntry(name="Ronald Acuña Jr.", adp=3.0, positions=("OF",)),),
            "source1",
        )
        source2 = MockADPSource(
            # Different representation of same player
            (ADPEntry(name="Ronald Acuna Jr", adp=5.0, positions=("OF",)),),
            "source2",
        )
        composite = CompositeADPSource([source1, source2])

        result = composite.fetch_adp()

        assert len(result.entries) == 1
        assert result.entries[0].adp == 4.0  # Average of 3.0 and 5.0

    def test_preserves_original_name(self) -> None:
        """Test that the first seen name variant is preserved."""
        source1 = MockADPSource(
            (ADPEntry(name="Ronald Acuña Jr.", adp=3.0, positions=("OF",)),),
            "source1",
        )
        source2 = MockADPSource(
            (ADPEntry(name="Ronald Acuna Jr", adp=5.0, positions=("OF",)),),
            "source2",
        )
        composite = CompositeADPSource([source1, source2])

        result = composite.fetch_adp()

        # Should preserve the first name seen
        assert result.entries[0].name == "Ronald Acuña Jr."

    def test_merges_positions(self) -> None:
        """Test that positions are merged from all sources."""
        source1 = MockADPSource(
            (ADPEntry(name="Shohei Ohtani", adp=2.0, positions=("DH",)),),
            "source1",
        )
        source2 = MockADPSource(
            (ADPEntry(name="Shohei Ohtani", adp=3.0, positions=("SP",)),),
            "source2",
        )
        composite = CompositeADPSource([source1, source2])

        result = composite.fetch_adp()

        assert len(result.entries) == 1
        # Should have both positions
        assert set(result.entries[0].positions) == {"DH", "SP"}

    def test_sorted_by_adp(self) -> None:
        """Test that results are sorted by ADP."""
        source = MockADPSource(
            (
                ADPEntry(name="Player C", adp=30.0, positions=("OF",)),
                ADPEntry(name="Player A", adp=10.0, positions=("1B",)),
                ADPEntry(name="Player B", adp=20.0, positions=("SS",)),
            ),
            "mock",
        )
        composite = CompositeADPSource([source])

        result = composite.fetch_adp()

        adps = [e.adp for e in result.entries]
        assert adps == [10.0, 20.0, 30.0]

    def test_averages_percent_drafted(self) -> None:
        """Test that percent_drafted is averaged when present."""
        source1 = MockADPSource(
            (ADPEntry(name="Mike Trout", adp=1.0, positions=("OF",), percent_drafted=90.0),),
            "source1",
        )
        source2 = MockADPSource(
            (ADPEntry(name="Mike Trout", adp=2.0, positions=("OF",), percent_drafted=80.0),),
            "source2",
        )
        composite = CompositeADPSource([source1, source2])

        result = composite.fetch_adp()

        assert result.entries[0].percent_drafted == 85.0

    def test_handles_none_percent_drafted(self) -> None:
        """Test that None percent_drafted is handled correctly."""
        source1 = MockADPSource(
            (ADPEntry(name="Mike Trout", adp=1.0, positions=("OF",), percent_drafted=90.0),),
            "source1",
        )
        source2 = MockADPSource(
            (ADPEntry(name="Mike Trout", adp=2.0, positions=("OF",), percent_drafted=None),),
            "source2",
        )
        composite = CompositeADPSource([source1, source2])

        result = composite.fetch_adp()

        # Should only use the non-None value
        assert result.entries[0].percent_drafted == 90.0

    def test_empty_sources_list(self) -> None:
        """Test handling of empty sources list."""
        composite = CompositeADPSource([])
        result = composite.fetch_adp()

        assert len(result.entries) == 0
        assert result.source == "composite"
