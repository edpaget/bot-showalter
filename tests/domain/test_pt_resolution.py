import pytest

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.pt_resolution import resolve_playing_time


def _proj(player_id: int, system: str, player_type: str, stats: dict) -> Projection:
    return Projection(
        player_id=player_id,
        season=2025,
        system=system,
        version="latest",
        player_type=player_type,
        stat_json=stats,
    )


def _fake_fetch(projections: list[Projection]):
    """Return a fetch_projections callable backed by the given list."""

    def fetch(season: int, system: str | None = None) -> list[Projection]:
        result = [p for p in projections if p.season == season]
        if system is not None:
            result = [p for p in result if p.system == system]
        return result

    return fetch


class TestResolveNative:
    def test_native_returns_none(self) -> None:
        result = resolve_playing_time("native", 2025, _fake_fetch([]))
        assert result is None


class TestResolveConsensusDefault:
    def test_consensus_default_uses_steamer_zips(self) -> None:
        projs = [
            _proj(1, "steamer", "batter", {"pa": 600.0}),
            _proj(1, "zips", "batter", {"pa": 500.0}),
        ]
        result = resolve_playing_time("consensus", 2025, _fake_fetch(projs))
        assert result is not None
        assert result.batting_pt[1] == pytest.approx(550.0)


class TestResolveConsensusInline:
    def test_consensus_inline_three_systems(self) -> None:
        projs = [
            _proj(1, "steamer", "batter", {"pa": 600.0}),
            _proj(1, "zips", "batter", {"pa": 500.0}),
            _proj(1, "atc", "batter", {"pa": 400.0}),
        ]
        result = resolve_playing_time("consensus:steamer,zips,atc", 2025, _fake_fetch(projs))
        assert result is not None
        assert result.batting_pt[1] == pytest.approx(500.0)

    def test_consensus_colon_with_spaces(self) -> None:
        projs = [
            _proj(1, "steamer", "batter", {"pa": 600.0}),
            _proj(1, "zips", "batter", {"pa": 400.0}),
        ]
        result = resolve_playing_time("consensus: steamer , zips ", 2025, _fake_fetch(projs))
        assert result is not None
        assert result.batting_pt[1] == pytest.approx(500.0)


class TestResolveSingleSystem:
    def test_single_system_name(self) -> None:
        projs = [
            _proj(1, "steamer", "batter", {"pa": 550.0}),
            _proj(2, "steamer", "pitcher", {"ip": 180.0}),
        ]
        result = resolve_playing_time("steamer", 2025, _fake_fetch(projs))
        assert result is not None
        assert result.batting_pt[1] == 550.0
        assert result.pitching_pt[2] == 180.0

    def test_playing_time_model_system(self) -> None:
        projs = [
            _proj(1, "playing-time-model", "batter", {"pa": 475.0}),
        ]
        result = resolve_playing_time("playing-time-model", 2025, _fake_fetch(projs))
        assert result is not None
        assert result.batting_pt[1] == 475.0


class TestResolveErrors:
    def test_unknown_system_empty_returns_empty_lookup(self) -> None:
        """System with no projections returns empty lookup (fallback to native)."""
        result = resolve_playing_time("no-such-system", 2025, _fake_fetch([]))
        assert result is not None
        assert result.batting_pt == {}
        assert result.pitching_pt == {}

    def test_unknown_system_warns(self) -> None:
        """System with no projections emits a warning."""
        with pytest.warns(UserWarning, match="no-such-system"):
            resolve_playing_time("no-such-system", 2025, _fake_fetch([]))

    def test_unknown_system_warns_with_available_systems(self) -> None:
        """Warning for unknown system includes available PT systems."""
        projs = [
            _proj(1, "steamer", "batter", {"pa": 600.0}),
        ]
        with pytest.warns(UserWarning, match="Available PT systems: steamer") as warnings:
            resolve_playing_time("no-such-system", 2025, _fake_fetch(projs))
        assert len(warnings) == 1

    def test_consensus_missing_system_warns(self) -> None:
        """Consensus with one missing system warns about that specific system."""
        projs = [
            _proj(1, "steamer", "batter", {"pa": 600.0}),
            _proj(1, "zips", "batter", {"pa": 500.0}),
        ]
        with pytest.warns(UserWarning, match="atc") as warnings:
            resolve_playing_time("consensus:steamer,zips,atc", 2025, _fake_fetch(projs))
        # Should warn about atc specifically
        assert any("atc" in str(w.message) for w in warnings)
        assert any("Available PT systems:" in str(w.message) for w in warnings)
