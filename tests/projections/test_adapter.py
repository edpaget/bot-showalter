"""Tests for ExternalProjectionAdapter player ID resolution."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper
from fantasy_baseball_manager.projections.adapter import ExternalProjectionAdapter


@pytest.fixture
def dummy_sources() -> tuple[Mock, Mock]:
    """Create dummy batting/pitching DataSources (unused for ID resolution tests)."""
    return Mock(), Mock()


class TestResolvePlayerId:
    """Tests for _resolve_player_id behaviour."""

    def test_prefers_fangraphs_id_when_both_present(self, dummy_sources: tuple[Mock, Mock]) -> None:
        adapter = ExternalProjectionAdapter(*dummy_sources)
        result = adapter._resolve_player_id("12345", "9876")
        assert result == "9876"

    def test_fangraphs_id_used_when_mlbam_is_none(self, dummy_sources: tuple[Mock, Mock]) -> None:
        adapter = ExternalProjectionAdapter(*dummy_sources)
        result = adapter._resolve_player_id(None, "9876")
        assert result == "9876"

    def test_mlbam_fallback_when_fangraphs_id_empty(self, dummy_sources: tuple[Mock, Mock]) -> None:
        adapter = ExternalProjectionAdapter(*dummy_sources)
        result = adapter._resolve_player_id("12345", "")
        assert result == "12345"

    def test_mapper_resolves_fangraphs_from_mlbam(self, dummy_sources: tuple[Mock, Mock]) -> None:
        mapper = Mock(spec=PlayerIdMapper)
        mapper.mlbam_to_fangraphs.return_value = "fg999"
        adapter = ExternalProjectionAdapter(*dummy_sources, id_mapper=mapper)
        result = adapter._resolve_player_id("12345", "")
        assert result == "fg999"
        mapper.mlbam_to_fangraphs.assert_called_once_with("12345")

    def test_mapper_miss_falls_back_to_mlbam(self, dummy_sources: tuple[Mock, Mock]) -> None:
        mapper = Mock(spec=PlayerIdMapper)
        mapper.mlbam_to_fangraphs.return_value = None
        adapter = ExternalProjectionAdapter(*dummy_sources, id_mapper=mapper)
        result = adapter._resolve_player_id("12345", "")
        assert result == "12345"

    def test_empty_string_when_both_missing(self, dummy_sources: tuple[Mock, Mock]) -> None:
        adapter = ExternalProjectionAdapter(*dummy_sources)
        result = adapter._resolve_player_id(None, "")
        assert result == ""
