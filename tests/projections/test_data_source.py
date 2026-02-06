"""Tests for projection DataSource implementations."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.player.identity import Player
from fantasy_baseball_manager.projections.data_source import (
    BattingProjectionDataSource,
    PitchingProjectionDataSource,
    create_batting_projection_source,
    create_pitching_projection_source,
)
from fantasy_baseball_manager.projections.models import (
    BattingProjection,
    PitchingProjection,
    ProjectionData,
    ProjectionSystem,
)


@pytest.fixture
def sample_batting_projection() -> BattingProjection:
    """Create a sample batting projection."""
    return BattingProjection(
        player_id="15640",
        mlbam_id="592450",
        name="Aaron Judge",
        team="NYY",
        position="OF",
        g=155,
        pa=635,
        ab=512,
        h=146,
        singles=78,
        doubles=24,
        triples=1,
        hr=43,
        r=110,
        rbi=104,
        sb=9,
        cs=2,
        bb=112,
        so=152,
        hbp=7,
        sf=4,
        sh=2,
        obp=0.417,
        slg=0.583,
        ops=1.000,
        woba=0.420,
        war=6.7,
    )


@pytest.fixture
def sample_pitching_projection() -> PitchingProjection:
    """Create a sample pitching projection."""
    return PitchingProjection(
        player_id="22267",
        mlbam_id="669373",
        name="Tarik Skubal",
        team="DET",
        g=32,
        gs=32,
        ip=199.8,
        w=14,
        l=8,
        sv=0,
        hld=0,
        so=243,
        bb=45,
        hbp=7,
        h=150,
        er=62,
        hr=20,
        era=2.80,
        whip=1.02,
        fip=3.10,
        war=5.2,
    )


@pytest.fixture
def sample_projection_data(
    sample_batting_projection: BattingProjection,
    sample_pitching_projection: PitchingProjection,
) -> ProjectionData:
    """Create sample projection data."""
    return ProjectionData(
        batting=(sample_batting_projection,),
        pitching=(sample_pitching_projection,),
        system=ProjectionSystem.STEAMER,
        fetched_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_projection_source(sample_projection_data: ProjectionData) -> Mock:
    """Create a mock ProjectionSource."""
    source = Mock()
    source.fetch_projections.return_value = sample_projection_data
    return source


class TestBattingProjectionDataSource:
    """Tests for BattingProjectionDataSource."""

    def test_all_players_query_returns_batting_projections(
        self,
        mock_projection_source: Mock,
        sample_batting_projection: BattingProjection,
    ) -> None:
        """ALL_PLAYERS query returns Ok with list of BattingProjection."""
        data_source = BattingProjectionDataSource(mock_projection_source)

        result = data_source(ALL_PLAYERS)

        assert result.is_ok()
        projections = result.unwrap()
        assert len(projections) == 1
        assert projections[0] == sample_batting_projection

    def test_single_player_query_returns_error(
        self,
        mock_projection_source: Mock,
    ) -> None:
        """Single Player query returns Err."""
        data_source = BattingProjectionDataSource(mock_projection_source)
        player = Player(name="Aaron Judge", yahoo_id="9877")

        result = data_source(player)

        assert result.is_err()
        error = result.unwrap_err()
        assert isinstance(error, DataSourceError)
        assert "ALL_PLAYERS" in error.message

    def test_player_list_query_returns_error(
        self,
        mock_projection_source: Mock,
    ) -> None:
        """list[Player] query returns Err."""
        data_source = BattingProjectionDataSource(mock_projection_source)
        players = [Player(name="Aaron Judge", yahoo_id="9877")]

        result = data_source(players)

        assert result.is_err()
        error = result.unwrap_err()
        assert isinstance(error, DataSourceError)
        assert "ALL_PLAYERS" in error.message

    def test_source_exception_returns_error(
        self,
        mock_projection_source: Mock,
    ) -> None:
        """Exception from source returns Err with cause."""
        mock_projection_source.fetch_projections.side_effect = RuntimeError("API error")
        data_source = BattingProjectionDataSource(mock_projection_source)

        result = data_source(ALL_PLAYERS)

        assert result.is_err()
        error = result.unwrap_err()
        assert isinstance(error, DataSourceError)
        assert "Failed to fetch batting projections" in error.message
        assert isinstance(error.cause, RuntimeError)


class TestPitchingProjectionDataSource:
    """Tests for PitchingProjectionDataSource."""

    def test_all_players_query_returns_pitching_projections(
        self,
        mock_projection_source: Mock,
        sample_pitching_projection: PitchingProjection,
    ) -> None:
        """ALL_PLAYERS query returns Ok with list of PitchingProjection."""
        data_source = PitchingProjectionDataSource(mock_projection_source)

        result = data_source(ALL_PLAYERS)

        assert result.is_ok()
        projections = result.unwrap()
        assert len(projections) == 1
        assert projections[0] == sample_pitching_projection

    def test_single_player_query_returns_error(
        self,
        mock_projection_source: Mock,
    ) -> None:
        """Single Player query returns Err."""
        data_source = PitchingProjectionDataSource(mock_projection_source)
        player = Player(name="Tarik Skubal", yahoo_id="11234")

        result = data_source(player)

        assert result.is_err()
        error = result.unwrap_err()
        assert isinstance(error, DataSourceError)
        assert "ALL_PLAYERS" in error.message

    def test_player_list_query_returns_error(
        self,
        mock_projection_source: Mock,
    ) -> None:
        """list[Player] query returns Err."""
        data_source = PitchingProjectionDataSource(mock_projection_source)
        players = [Player(name="Tarik Skubal", yahoo_id="11234")]

        result = data_source(players)

        assert result.is_err()
        error = result.unwrap_err()
        assert isinstance(error, DataSourceError)
        assert "ALL_PLAYERS" in error.message

    def test_source_exception_returns_error(
        self,
        mock_projection_source: Mock,
    ) -> None:
        """Exception from source returns Err with cause."""
        mock_projection_source.fetch_projections.side_effect = RuntimeError("API error")
        data_source = PitchingProjectionDataSource(mock_projection_source)

        result = data_source(ALL_PLAYERS)

        assert result.is_err()
        error = result.unwrap_err()
        assert isinstance(error, DataSourceError)
        assert "Failed to fetch pitching projections" in error.message
        assert isinstance(error.cause, RuntimeError)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_batting_projection_source_returns_data_source(self) -> None:
        """create_batting_projection_source returns a BattingProjectionDataSource."""
        source = create_batting_projection_source()

        assert isinstance(source, BattingProjectionDataSource)

    def test_create_batting_projection_source_accepts_system(self) -> None:
        """create_batting_projection_source accepts projection system."""
        source = create_batting_projection_source(ProjectionSystem.ZIPS)

        assert isinstance(source, BattingProjectionDataSource)

    def test_create_pitching_projection_source_returns_data_source(self) -> None:
        """create_pitching_projection_source returns a PitchingProjectionDataSource."""
        source = create_pitching_projection_source()

        assert isinstance(source, PitchingProjectionDataSource)

    def test_create_pitching_projection_source_accepts_system(self) -> None:
        """create_pitching_projection_source accepts projection system."""
        source = create_pitching_projection_source(ProjectionSystem.ZIPS)

        assert isinstance(source, PitchingProjectionDataSource)
