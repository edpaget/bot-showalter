from collections.abc import Callable
from typing import Any

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.ingest.column_maps import (
    make_fg_projection_batting_mapper,
    make_fg_projection_pitching_mapper,
)


def _trout() -> Player:
    return Player(
        name_first="Mike",
        name_last="Trout",
        id=1,
        mlbam_id=545361,
        fangraphs_id=10155,
    )


def _fg_projection_batting_row(
    *,
    idfg: int | float | str = 10155,
    mlbamid: int | float = 545361,
    pa: int = 600,
    ab: int = 530,
    h: int = 160,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 35,
    rbi: int = 90,
    r: int = 100,
    sb: int = 15,
    cs: int = 3,
    bb: int = 60,
    so: int = 120,
    avg: float = 0.302,
    obp: float = 0.395,
    slg: float = 0.575,
    ops: float = 0.970,
    woba: float = 0.410,
    wrc_plus: float = 170.0,
    war: float = 8.5,
) -> dict[str, Any]:
    return {
        "PlayerId": idfg,
        "MLBAMID": mlbamid,
        "PA": pa,
        "AB": ab,
        "H": h,
        "2B": doubles,
        "3B": triples,
        "HR": hr,
        "RBI": rbi,
        "R": r,
        "SB": sb,
        "CS": cs,
        "BB": bb,
        "SO": so,
        "AVG": avg,
        "OBP": obp,
        "SLG": slg,
        "OPS": ops,
        "wOBA": woba,
        "wRC+": wrc_plus,
        "WAR": war,
    }


class TestFgProjectionBattingMapper:
    def test_complete_row(self) -> None:
        mapper = make_fg_projection_batting_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        result = mapper(_fg_projection_batting_row())

        assert result is not None
        assert result.player_id == 1
        assert result.season == 2025
        assert result.system == "steamer"
        assert result.version == "2025.1"
        assert result.player_type == "batter"
        assert result.stat_json["hr"] == 35
        assert result.stat_json["avg"] == 0.302
        assert result.stat_json["pa"] == 600
        assert result.stat_json["war"] == 8.5
        assert result.stat_json["sb"] == 15
        assert result.stat_json["woba"] == 0.410
        assert result.stat_json["wrc_plus"] == 170.0

    def test_unknown_id_returns_none(self) -> None:
        mapper = make_fg_projection_batting_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        result = mapper(_fg_projection_batting_row(idfg=99999, mlbamid=888888))
        assert result is None

    def test_nan_id_returns_none(self) -> None:
        mapper = make_fg_projection_batting_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        result = mapper(_fg_projection_batting_row(idfg=float("nan"), mlbamid=float("nan")))
        assert result is None

    def test_nan_stat_excluded_from_json(self) -> None:
        mapper = make_fg_projection_batting_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        result = mapper(_fg_projection_batting_row(war=float("nan")))
        assert result is not None
        assert "war" not in result.stat_json

    def test_default_source_type_is_first_party(self) -> None:
        mapper = make_fg_projection_batting_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        result = mapper(_fg_projection_batting_row())
        assert result is not None
        assert result.source_type == "first_party"

    def test_third_party_source_type(self) -> None:
        mapper = make_fg_projection_batting_mapper(
            [_trout()], season=2025, system="steamer", version="2025.1", source_type="third_party"
        )
        result = mapper(_fg_projection_batting_row())
        assert result is not None
        assert result.source_type == "third_party"

    def test_tier2_batting_stats_in_stat_json(self) -> None:
        mapper = make_fg_projection_batting_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        row = _fg_projection_batting_row()
        row["1B"] = 76
        row["G"] = 141
        row["ISO"] = 0.273
        row["BABIP"] = 0.320
        row["BB%"] = 0.10
        row["K%"] = 0.20
        result = mapper(row)
        assert result is not None
        assert result.stat_json["singles"] == 76
        assert result.stat_json["g"] == 141
        assert result.stat_json["iso"] == 0.273
        assert result.stat_json["babip"] == 0.320
        assert result.stat_json["bb_pct"] == 0.10
        assert result.stat_json["k_pct"] == 0.20

    def test_works_with_zips(self) -> None:
        mapper = make_fg_projection_batting_mapper([_trout()], season=2025, system="zips", version="2025.1")
        result = mapper(_fg_projection_batting_row())
        assert result is not None
        assert result.system == "zips"


def _fg_projection_pitching_row(
    *,
    idfg: int | float | str = 10155,
    mlbamid: int | float = 545361,
    w: int = 15,
    losses: int = 8,
    g: int = 32,
    gs: int = 32,
    sv: int = 0,
    h: int = 140,
    er: int = 55,
    hr: int = 18,
    bb: int = 45,
    so: int = 220,
    era: float = 2.85,
    ip: float = 190.1,
    whip: float = 0.97,
    k_per_9: float = 10.4,
    bb_per_9: float = 2.1,
    fip: float = 3.10,
    war: float = 6.0,
) -> dict[str, Any]:
    return {
        "PlayerId": idfg,
        "MLBAMID": mlbamid,
        "W": w,
        "L": losses,
        "G": g,
        "GS": gs,
        "SV": sv,
        "H": h,
        "ER": er,
        "HR": hr,
        "BB": bb,
        "SO": so,
        "ERA": era,
        "IP": ip,
        "WHIP": whip,
        "K/9": k_per_9,
        "BB/9": bb_per_9,
        "FIP": fip,
        "WAR": war,
    }


class TestFgProjectionPitchingMapper:
    def test_complete_row(self) -> None:
        mapper = make_fg_projection_pitching_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        result = mapper(_fg_projection_pitching_row())

        assert result is not None
        assert result.player_id == 1
        assert result.season == 2025
        assert result.system == "steamer"
        assert result.version == "2025.1"
        assert result.player_type == "pitcher"
        assert result.stat_json["w"] == 15
        assert result.stat_json["l"] == 8
        assert result.stat_json["era"] == 2.85
        assert result.stat_json["so"] == 220
        assert result.stat_json["ip"] == 190.1
        assert result.stat_json["war"] == 6.0
        assert result.stat_json["k_per_9"] == 10.4

    def test_unknown_id_returns_none(self) -> None:
        mapper = make_fg_projection_pitching_mapper([_trout()], season=2025, system="zips", version="2025.1")
        result = mapper(_fg_projection_pitching_row(idfg=99999, mlbamid=888888))
        assert result is None

    def test_default_source_type_is_first_party(self) -> None:
        mapper = make_fg_projection_pitching_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        result = mapper(_fg_projection_pitching_row())
        assert result is not None
        assert result.source_type == "first_party"

    def test_third_party_source_type(self) -> None:
        mapper = make_fg_projection_pitching_mapper(
            [_trout()], season=2025, system="steamer", version="2025.1", source_type="third_party"
        )
        result = mapper(_fg_projection_pitching_row())
        assert result is not None
        assert result.source_type == "third_party"

    def test_nan_stat_excluded_from_json(self) -> None:
        mapper = make_fg_projection_pitching_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        result = mapper(_fg_projection_pitching_row(war=float("nan")))
        assert result is not None
        assert "war" not in result.stat_json

    def test_tier2_pitching_stats_in_stat_json(self) -> None:
        mapper = make_fg_projection_pitching_mapper([_trout()], season=2025, system="steamer", version="2025.1")
        row = _fg_projection_pitching_row()
        row["QS"] = 20
        row["TBF"] = 750
        row["K%"] = 0.28
        row["BABIP"] = 0.290
        row["GB%"] = 0.45
        result = mapper(row)
        assert result is not None
        assert result.stat_json["qs"] == 20
        assert result.stat_json["tbf"] == 750
        assert result.stat_json["k_pct"] == 0.28
        assert result.stat_json["babip"] == 0.290
        assert result.stat_json["gb_pct"] == 0.45


class TestResolvePlayerIdFallback:
    def _make_mapper(self) -> Callable[[dict[str, Any]], Projection | None]:
        return make_fg_projection_batting_mapper([_trout()], season=2025, system="steamer", version="2025.1")

    def test_numeric_fg_id_resolves(self) -> None:
        mapper = self._make_mapper()
        result = mapper(_fg_projection_batting_row(idfg=10155, mlbamid=545361))
        assert result is not None
        assert result.player_id == 1

    def test_sa_prefix_with_valid_mlbamid(self) -> None:
        mapper = self._make_mapper()
        result = mapper(_fg_projection_batting_row(idfg="sa3025154", mlbamid=545361))
        assert result is not None
        assert result.player_id == 1

    def test_sa_prefix_without_mlbamid(self) -> None:
        mapper = self._make_mapper()
        result = mapper(_fg_projection_batting_row(idfg="sa3025154", mlbamid=float("nan")))
        assert result is None

    def test_numeric_not_in_fg_falls_back_to_mlbam(self) -> None:
        mapper = self._make_mapper()
        result = mapper(_fg_projection_batting_row(idfg=99999, mlbamid=545361))
        assert result is not None
        assert result.player_id == 1

    def test_both_lookups_fail(self) -> None:
        mapper = self._make_mapper()
        result = mapper(_fg_projection_batting_row(idfg=99999, mlbamid=888888))
        assert result is None
