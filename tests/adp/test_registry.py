"""Tests for ADP provider registry."""

import pytest

from fantasy_baseball_manager.adp.models import ADPData
from fantasy_baseball_manager.adp.registry import (
    get_source,
    list_sources,
    register_source,
    reset_registry,
)


class MockADPSource:
    """Mock ADP source for testing."""

    def __init__(self, source_name: str = "mock") -> None:
        self._source_name = source_name

    def fetch_adp(self) -> ADPData:
        from datetime import UTC, datetime

        return ADPData(
            entries=(),
            fetched_at=datetime.now(UTC),
            source=self._source_name,
        )


class TestRegistry:
    """Tests for the ADP provider registry."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_registry()

    def test_register_and_get_source(self) -> None:
        """Test registering and retrieving a source."""
        register_source("mock", lambda: MockADPSource("mock"))
        source = get_source("mock")

        assert source is not None
        result = source.fetch_adp()
        assert result.source == "mock"

    def test_get_unknown_source_raises(self) -> None:
        """Test that getting unknown source raises KeyError."""
        with pytest.raises(KeyError, match="Unknown ADP source"):
            get_source("nonexistent")

    def test_list_sources(self) -> None:
        """Test listing registered sources."""
        register_source("source1", lambda: MockADPSource("source1"))
        register_source("source2", lambda: MockADPSource("source2"))

        sources = list_sources()

        assert "source1" in sources
        assert "source2" in sources

    def test_list_sources_after_reset_has_defaults(self) -> None:
        """Test that list_sources auto-registers defaults after reset."""
        # After reset, list_sources will trigger default registration
        sources = list_sources()
        assert "yahoo" in sources
        assert "espn" in sources

    def test_register_overwrites_existing(self) -> None:
        """Test that registering same name overwrites."""
        register_source("mock", lambda: MockADPSource("old"))
        register_source("mock", lambda: MockADPSource("new"))

        source = get_source("mock")
        result = source.fetch_adp()
        assert result.source == "new"


class TestDefaultProviders:
    """Tests for auto-registered providers."""

    def test_yahoo_registered(self) -> None:
        """Test that yahoo is auto-registered."""
        from fantasy_baseball_manager.adp.registry import _ensure_defaults_registered

        reset_registry()
        _ensure_defaults_registered()

        sources = list_sources()
        assert "yahoo" in sources

    def test_espn_registered(self) -> None:
        """Test that espn is auto-registered."""
        from fantasy_baseball_manager.adp.registry import _ensure_defaults_registered

        reset_registry()
        _ensure_defaults_registered()

        sources = list_sources()
        assert "espn" in sources

    def test_get_yahoo_source(self) -> None:
        """Test getting yahoo source returns YahooADPScraper."""
        from fantasy_baseball_manager.adp.registry import _ensure_defaults_registered
        from fantasy_baseball_manager.adp.scraper import YahooADPScraper

        reset_registry()
        _ensure_defaults_registered()

        source = get_source("yahoo")
        assert isinstance(source, YahooADPScraper)

    def test_get_espn_source(self) -> None:
        """Test getting espn source returns ESPNADPScraper."""
        from fantasy_baseball_manager.adp.espn_scraper import ESPNADPScraper
        from fantasy_baseball_manager.adp.registry import _ensure_defaults_registered

        reset_registry()
        _ensure_defaults_registered()

        source = get_source("espn")
        assert isinstance(source, ESPNADPScraper)
