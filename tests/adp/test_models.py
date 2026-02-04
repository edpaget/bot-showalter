"""Tests for ADP models."""

from datetime import UTC, datetime

import pytest

from fantasy_baseball_manager.adp.models import ADPData, ADPEntry


class TestADPEntry:
    """Tests for ADPEntry dataclass."""

    def test_construction(self) -> None:
        """Test basic construction of ADPEntry."""
        entry = ADPEntry(
            name="Mike Trout",
            adp=1.5,
            positions=("OF",),
            percent_drafted=99.8,
        )
        assert entry.name == "Mike Trout"
        assert entry.adp == 1.5
        assert entry.positions == ("OF",)
        assert entry.percent_drafted == 99.8

    def test_construction_multiple_positions(self) -> None:
        """Test ADPEntry with multiple positions."""
        entry = ADPEntry(
            name="Shohei Ohtani",
            adp=2.0,
            positions=("DH", "SP"),
        )
        assert entry.positions == ("DH", "SP")

    def test_percent_drafted_optional(self) -> None:
        """Test that percent_drafted is optional."""
        entry = ADPEntry(
            name="Player Name",
            adp=10.0,
            positions=("1B",),
        )
        assert entry.percent_drafted is None

    def test_immutability(self) -> None:
        """Test that ADPEntry is immutable (frozen)."""
        entry = ADPEntry(
            name="Mike Trout",
            adp=1.5,
            positions=("OF",),
        )
        with pytest.raises(AttributeError):
            entry.name = "Other Player"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Test that identical ADPEntry instances are equal."""
        entry1 = ADPEntry(name="Player", adp=5.0, positions=("SS",))
        entry2 = ADPEntry(name="Player", adp=5.0, positions=("SS",))
        assert entry1 == entry2

    def test_hashable(self) -> None:
        """Test that ADPEntry is hashable for use in sets/dicts."""
        entry = ADPEntry(name="Player", adp=5.0, positions=("SS",))
        entry_set = {entry}
        assert entry in entry_set


class TestADPData:
    """Tests for ADPData dataclass."""

    def test_construction(self) -> None:
        """Test basic construction of ADPData."""
        entries = (
            ADPEntry(name="Player 1", adp=1.0, positions=("OF",)),
            ADPEntry(name="Player 2", adp=2.0, positions=("1B",)),
        )
        fetched_at = datetime(2025, 3, 15, 12, 0, 0, tzinfo=UTC)
        data = ADPData(entries=entries, fetched_at=fetched_at)

        assert len(data.entries) == 2
        assert data.entries[0].name == "Player 1"
        assert data.fetched_at == fetched_at
        assert data.source == "yahoo"

    def test_custom_source(self) -> None:
        """Test ADPData with custom source."""
        data = ADPData(
            entries=(),
            fetched_at=datetime.now(UTC),
            source="espn",
        )
        assert data.source == "espn"

    def test_immutability(self) -> None:
        """Test that ADPData is immutable (frozen)."""
        data = ADPData(
            entries=(),
            fetched_at=datetime.now(UTC),
        )
        with pytest.raises(AttributeError):
            data.source = "other"  # type: ignore[misc]

    def test_empty_entries(self) -> None:
        """Test ADPData with empty entries."""
        data = ADPData(
            entries=(),
            fetched_at=datetime.now(UTC),
        )
        assert len(data.entries) == 0
