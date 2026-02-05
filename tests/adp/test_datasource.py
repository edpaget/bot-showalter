"""Tests for new-style DataSource[ADPEntry] implementations."""

from collections.abc import Generator
from typing import cast

import pytest

from fantasy_baseball_manager.adp.composite import (
    CompositeADPDataSource,
    create_composite_adp_source,
)
from fantasy_baseball_manager.adp.espn_scraper import (
    ESPNADPDataSource,
    create_espn_adp_source,
)
from fantasy_baseball_manager.adp.models import ADPEntry
from fantasy_baseball_manager.adp.registry import (
    get_datasource,
    list_datasources,
    register_datasource,
    reset_registry,
)
from fantasy_baseball_manager.adp.scraper import YahooADPDataSource, create_yahoo_adp_source
from fantasy_baseball_manager.context import init_context, reset_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSource, DataSourceError
from fantasy_baseball_manager.player.identity import Player
from fantasy_baseball_manager.result import Err, Ok


def _mock_adp_source(entries: list[ADPEntry], should_fail: bool = False) -> DataSource[ADPEntry]:
    """Create a mock DataSource[ADPEntry] for testing."""

    def source(
        query: type[ALL_PLAYERS] | Player | list[Player],
    ) -> Ok[list[ADPEntry]] | Ok[ADPEntry] | Err[DataSourceError]:
        if should_fail:
            return Err(DataSourceError("Mock failure"))
        if query is not ALL_PLAYERS:
            return Err(DataSourceError("Only ALL_PLAYERS queries supported"))
        return Ok(entries)

    return cast("DataSource[ADPEntry]", source)


@pytest.fixture(autouse=True)
def setup_context() -> Generator[None]:
    """Initialize context before each test."""
    init_context(year=2025)
    yield
    reset_context()


class TestYahooADPDataSource:
    """Tests for YahooADPDataSource."""

    def test_only_supports_all_players_query(self) -> None:
        """Test that only ALL_PLAYERS queries are supported."""
        source = YahooADPDataSource()
        player = Player(name="Test Player", yahoo_id="12345")

        result = source(player)

        assert result.is_err()
        err = cast("DataSourceError", result.unwrap_err())
        assert "Only ALL_PLAYERS queries supported" in err.message

    def test_create_factory_returns_datasource(self) -> None:
        """Test that create_yahoo_adp_source returns a DataSource."""
        source = create_yahoo_adp_source()
        assert source is not None


class TestESPNADPDataSource:
    """Tests for ESPNADPDataSource."""

    def test_only_supports_all_players_query(self) -> None:
        """Test that only ALL_PLAYERS queries are supported."""
        source = ESPNADPDataSource()
        player = Player(name="Test Player", yahoo_id="12345")

        result = source(player)

        assert result.is_err()
        err = cast("DataSourceError", result.unwrap_err())
        assert "Only ALL_PLAYERS queries supported" in err.message

    def test_create_factory_returns_datasource(self) -> None:
        """Test that create_espn_adp_source returns a DataSource."""
        source = create_espn_adp_source()
        assert source is not None


class TestCompositeADPDataSource:
    """Tests for CompositeADPDataSource."""

    def test_single_source_passthrough(self) -> None:
        """Test that single source entries pass through."""
        entries = [
            ADPEntry(name="Mike Trout", adp=1.5, positions=("OF",)),
            ADPEntry(name="Shohei Ohtani", adp=2.3, positions=("DH", "SP")),
        ]
        mock_source = _mock_adp_source(entries)
        composite = CompositeADPDataSource([mock_source])

        result = composite(ALL_PLAYERS)

        assert result.is_ok()
        result_entries = result.unwrap()
        assert len(result_entries) == 2
        assert result_entries[0].name == "Mike Trout"
        assert result_entries[0].adp == 1.5

    def test_averages_adp_across_sources(self) -> None:
        """Test that ADP is averaged across multiple sources."""
        source1 = _mock_adp_source([ADPEntry(name="Mike Trout", adp=1.0, positions=("OF",))])
        source2 = _mock_adp_source([ADPEntry(name="Mike Trout", adp=3.0, positions=("OF",))])
        composite = CompositeADPDataSource([source1, source2])

        result = composite(ALL_PLAYERS)

        assert result.is_ok()
        entries = result.unwrap()
        assert len(entries) == 1
        assert entries[0].name == "Mike Trout"
        assert entries[0].adp == 2.0  # Average of 1.0 and 3.0

    def test_handles_players_in_only_one_source(self) -> None:
        """Test players appearing in only one source are included."""
        source1 = _mock_adp_source(
            [
                ADPEntry(name="Mike Trout", adp=1.0, positions=("OF",)),
                ADPEntry(name="Only Source 1", adp=50.0, positions=("1B",)),
            ]
        )
        source2 = _mock_adp_source(
            [
                ADPEntry(name="Mike Trout", adp=2.0, positions=("OF",)),
                ADPEntry(name="Only Source 2", adp=60.0, positions=("SS",)),
            ]
        )
        composite = CompositeADPDataSource([source1, source2])

        result = composite(ALL_PLAYERS)

        assert result.is_ok()
        entries = result.unwrap()
        names = {e.name for e in entries}
        assert names == {"Mike Trout", "Only Source 1", "Only Source 2"}

    def test_normalizes_names_for_matching(self) -> None:
        """Test that names are normalized for matching across sources."""
        source1 = _mock_adp_source([ADPEntry(name="Ronald Acuña Jr.", adp=3.0, positions=("OF",))])
        source2 = _mock_adp_source([ADPEntry(name="Ronald Acuna Jr", adp=5.0, positions=("OF",))])
        composite = CompositeADPDataSource([source1, source2])

        result = composite(ALL_PLAYERS)

        assert result.is_ok()
        entries = result.unwrap()
        assert len(entries) == 1
        assert entries[0].adp == 4.0  # Average of 3.0 and 5.0

    def test_merges_positions(self) -> None:
        """Test that positions are merged from all sources."""
        source1 = _mock_adp_source([ADPEntry(name="Shohei Ohtani", adp=2.0, positions=("DH",))])
        source2 = _mock_adp_source([ADPEntry(name="Shohei Ohtani", adp=3.0, positions=("SP",))])
        composite = CompositeADPDataSource([source1, source2])

        result = composite(ALL_PLAYERS)

        assert result.is_ok()
        entries = result.unwrap()
        assert len(entries) == 1
        assert set(entries[0].positions) == {"DH", "SP"}

    def test_sorted_by_adp(self) -> None:
        """Test that results are sorted by ADP."""
        source = _mock_adp_source(
            [
                ADPEntry(name="Player C", adp=30.0, positions=("OF",)),
                ADPEntry(name="Player A", adp=10.0, positions=("1B",)),
                ADPEntry(name="Player B", adp=20.0, positions=("SS",)),
            ]
        )
        composite = CompositeADPDataSource([source])

        result = composite(ALL_PLAYERS)

        assert result.is_ok()
        entries = result.unwrap()
        adps = [e.adp for e in entries]
        assert adps == [10.0, 20.0, 30.0]

    def test_empty_sources_list(self) -> None:
        """Test handling of empty sources list."""
        composite = CompositeADPDataSource([])

        result = composite(ALL_PLAYERS)

        assert result.is_ok()
        assert len(result.unwrap()) == 0

    def test_only_supports_all_players_query(self) -> None:
        """Test that only ALL_PLAYERS queries are supported."""
        composite = CompositeADPDataSource([])
        player = Player(name="Test Player", yahoo_id="12345")

        result = composite(player)

        assert result.is_err()
        err = cast("DataSourceError", result.unwrap_err())
        assert "Only ALL_PLAYERS queries supported" in err.message

    def test_propagates_source_errors(self) -> None:
        """Test that errors from sources are propagated."""
        failing_source = _mock_adp_source([], should_fail=True)
        composite = CompositeADPDataSource([failing_source])

        result = composite(ALL_PLAYERS)

        assert result.is_err()
        err = cast("DataSourceError", result.unwrap_err())
        assert "Mock failure" in err.message

    def test_create_factory_returns_datasource(self) -> None:
        """Test that create_composite_adp_source returns a DataSource."""
        source = create_composite_adp_source([])
        assert source is not None


class TestDataSourceRegistry:
    """Tests for the DataSource registry."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_registry()

    def test_register_and_get_datasource(self) -> None:
        """Test registering and retrieving a datasource."""
        mock_entries = [ADPEntry(name="Test", adp=1.0, positions=("OF",))]
        register_datasource("mock", lambda: _mock_adp_source(mock_entries))

        source = get_datasource("mock")
        result = source(ALL_PLAYERS)

        assert result.is_ok()
        assert len(result.unwrap()) == 1

    def test_get_unknown_datasource_raises(self) -> None:
        """Test that getting unknown datasource raises KeyError."""
        with pytest.raises(KeyError, match="Unknown ADP data source"):
            get_datasource("nonexistent")

    def test_list_datasources(self) -> None:
        """Test listing registered datasources."""
        mock_entries: list[ADPEntry] = []
        register_datasource("source1", lambda: _mock_adp_source(mock_entries))
        register_datasource("source2", lambda: _mock_adp_source(mock_entries))

        sources = list_datasources()

        assert "source1" in sources
        assert "source2" in sources

    def test_defaults_registered(self) -> None:
        """Test that default datasources are auto-registered."""
        sources = list_datasources()
        assert "yahoo" in sources
        assert "espn" in sources

    def test_get_yahoo_datasource(self) -> None:
        """Test getting yahoo datasource returns YahooADPDataSource."""
        source = get_datasource("yahoo")
        assert isinstance(source, YahooADPDataSource)

    def test_get_espn_datasource(self) -> None:
        """Test getting espn datasource returns ESPNADPDataSource."""
        source = get_datasource("espn")
        assert isinstance(source, ESPNADPDataSource)
