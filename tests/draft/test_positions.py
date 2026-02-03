from pathlib import Path

from fantasy_baseball_manager.draft.positions import (
    DEFAULT_ROSTER_CONFIG,
    infer_pitcher_role,
    load_positions_file,
    normalize_position,
)
from fantasy_baseball_manager.marcel.models import PitchingProjection


def _make_pitching_projection(
    gs: float = 30.0,
    g: float = 32.0,
) -> PitchingProjection:
    return PitchingProjection(
        player_id="p1",
        name="Test Pitcher",
        year=2025,
        age=28,
        ip=180.0,
        g=g,
        gs=gs,
        er=70.0,
        h=150.0,
        bb=50.0,
        so=200.0,
        hr=20.0,
        hbp=5.0,
        era=3.50,
        whip=1.11,
        w=0.0,
        nsvh=0.0,
    )


class TestNormalizePosition:
    def test_lf_to_of(self) -> None:
        assert normalize_position("LF") == "OF"

    def test_cf_to_of(self) -> None:
        assert normalize_position("CF") == "OF"

    def test_rf_to_of(self) -> None:
        assert normalize_position("RF") == "OF"

    def test_dh_to_util(self) -> None:
        assert normalize_position("DH") == "Util"

    def test_regular_position_unchanged(self) -> None:
        assert normalize_position("SS") == "SS"
        assert normalize_position("C") == "C"
        assert normalize_position("1B") == "1B"

    def test_strips_whitespace(self) -> None:
        assert normalize_position(" SS ") == "SS"


class TestInferPitcherRole:
    def test_starter_high_gs_ratio(self) -> None:
        proj = _make_pitching_projection(gs=30.0, g=32.0)
        assert infer_pitcher_role(proj) == "SP"

    def test_reliever_low_gs_ratio(self) -> None:
        proj = _make_pitching_projection(gs=2.0, g=60.0)
        assert infer_pitcher_role(proj) == "RP"

    def test_exactly_half_is_starter(self) -> None:
        proj = _make_pitching_projection(gs=16.0, g=32.0)
        assert infer_pitcher_role(proj) == "SP"

    def test_just_below_half_is_reliever(self) -> None:
        proj = _make_pitching_projection(gs=15.0, g=32.0)
        assert infer_pitcher_role(proj) == "RP"

    def test_zero_games_defaults_to_sp(self) -> None:
        proj = _make_pitching_projection(gs=0.0, g=0.0)
        assert infer_pitcher_role(proj) == "SP"


class TestLoadPositionsFile:
    def test_loads_basic_file(self, tmp_path: Path) -> None:
        f = tmp_path / "positions.csv"
        f.write_text("123,C/1B\n456,2B/SS\n")
        result = load_positions_file(f)
        assert result == {
            "123": ("C", "1B"),
            "456": ("2B", "SS"),
        }

    def test_normalizes_outfield(self, tmp_path: Path) -> None:
        f = tmp_path / "positions.csv"
        f.write_text("123,LF/CF/RF\n")
        result = load_positions_file(f)
        assert result == {"123": ("OF",)}

    def test_skips_comments(self, tmp_path: Path) -> None:
        f = tmp_path / "positions.csv"
        f.write_text("# comment\n123,SS\n")
        result = load_positions_file(f)
        assert result == {"123": ("SS",)}

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "positions.csv"
        f.write_text("123,SS\n\n456,OF\n")
        result = load_positions_file(f)
        assert len(result) == 2


class FakeIdMapper:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._yahoo_to_fg = mapping

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return self._yahoo_to_fg.get(yahoo_id)

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return None

    def fangraphs_to_mlbam(self, fangraphs_id: str) -> str | None:
        return None

    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
        return None


def _make_yahoo_player(
    player_id: int,
    name: str,
    eligible_positions: list[str],
    position_type: str = "B",
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "name": name,
        "eligible_positions": eligible_positions,
        "position_type": position_type,
    }


class TestYahooPositionSource:
    def test_fetches_and_maps_positions(self) -> None:
        from unittest.mock import MagicMock

        from fantasy_baseball_manager.draft.positions import YahooPositionSource

        league = MagicMock()
        league.taken_players.return_value = [
            _make_yahoo_player(101, "Mike Trout", ["CF", "DH"]),
        ]
        league.free_agents.side_effect = lambda pos_type: {
            "B": [_make_yahoo_player(102, "Freddie Freeman", ["1B"])],
            "P": [_make_yahoo_player(201, "Gerrit Cole", ["SP"], position_type="P")],
        }[pos_type]

        mapper = FakeIdMapper({"101": "fg1", "102": "fg2", "201": "fg3"})
        source = YahooPositionSource(league, mapper)
        result = source.fetch_positions()

        assert result["fg1"] == ("OF", "Util")
        assert result["fg2"] == ("1B",)
        assert result["fg3"] == ("SP",)

    def test_skips_players_without_fangraphs_id(self) -> None:
        from unittest.mock import MagicMock

        from fantasy_baseball_manager.draft.positions import YahooPositionSource

        league = MagicMock()
        league.taken_players.return_value = [
            _make_yahoo_player(101, "Mike Trout", ["CF"]),
            _make_yahoo_player(999, "Unknown Guy", ["SS"]),
        ]
        league.free_agents.return_value = []

        mapper = FakeIdMapper({"101": "fg1"})  # 999 not mapped
        source = YahooPositionSource(league, mapper)
        result = source.fetch_positions()

        assert "fg1" in result
        assert len(result) == 1

    def test_deduplicates_across_sources(self) -> None:
        from unittest.mock import MagicMock

        from fantasy_baseball_manager.draft.positions import YahooPositionSource

        league = MagicMock()
        # Same player in taken and free agents (shouldn't happen, but defensive)
        league.taken_players.return_value = [
            _make_yahoo_player(101, "Mike Trout", ["CF", "DH"]),
        ]
        league.free_agents.side_effect = lambda pos_type: {
            "B": [_make_yahoo_player(101, "Mike Trout", ["CF", "DH"])],
            "P": [],
        }[pos_type]

        mapper = FakeIdMapper({"101": "fg1"})
        source = YahooPositionSource(league, mapper)
        result = source.fetch_positions()

        assert len(result) == 1
        assert result["fg1"] == ("OF", "Util")

    def test_filters_bench_and_il_positions(self) -> None:
        from unittest.mock import MagicMock

        from fantasy_baseball_manager.draft.positions import YahooPositionSource

        league = MagicMock()
        league.taken_players.return_value = [
            _make_yahoo_player(101, "Player One", ["SS", "BN", "IL", "DL"]),
        ]
        league.free_agents.return_value = []

        mapper = FakeIdMapper({"101": "fg1"})
        source = YahooPositionSource(league, mapper)
        result = source.fetch_positions()

        assert result["fg1"] == ("SS",)


class TestDefaultRosterConfig:
    def test_has_standard_positions(self) -> None:
        position_names = {s.position for s in DEFAULT_ROSTER_CONFIG.slots}
        assert "C" in position_names
        assert "SP" in position_names
        assert "RP" in position_names
        assert "OF" in position_names

    def test_total_slots(self) -> None:
        total = sum(s.count for s in DEFAULT_ROSTER_CONFIG.slots)
        assert total == 28
