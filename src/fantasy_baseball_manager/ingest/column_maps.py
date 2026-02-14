import math
from collections.abc import Callable
from typing import Any

import pandas as pd

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.statcast_pitch import StatcastPitch


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    int_val = int(value)
    if int_val == -1:
        return None
    return int_val


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value)
    if s == "":
        return None
    return s


def _to_optional_int_stat(value: Any) -> int | None:
    """Like _to_optional_int but without the -1 sentinel handling."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def _build_fg_lookup(players: list[Player]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for p in players:
        if p.fangraphs_id is not None and p.id is not None:
            lookup[p.fangraphs_id] = p.id
    return lookup


def _resolve_fg_id(fg_lookup: dict[int, int], row: pd.Series) -> int | None:
    fg_id = row["IDfg"]
    if isinstance(fg_id, float) and math.isnan(fg_id):
        return None
    return fg_lookup.get(int(fg_id))


def make_fg_batting_mapper(
    players: list[Player],
) -> Callable[[pd.Series], BattingStats | None]:
    fg_lookup = _build_fg_lookup(players)

    def mapper(row: pd.Series) -> BattingStats | None:
        player_id = _resolve_fg_id(fg_lookup, row)
        if player_id is None:
            return None

        return BattingStats(
            player_id=player_id,
            season=int(row["Season"]),
            source="fangraphs",
            pa=_to_optional_int_stat(row["PA"]),
            ab=_to_optional_int_stat(row["AB"]),
            h=_to_optional_int_stat(row["H"]),
            doubles=_to_optional_int_stat(row["2B"]),
            triples=_to_optional_int_stat(row["3B"]),
            hr=_to_optional_int_stat(row["HR"]),
            rbi=_to_optional_int_stat(row["RBI"]),
            r=_to_optional_int_stat(row["R"]),
            sb=_to_optional_int_stat(row["SB"]),
            cs=_to_optional_int_stat(row["CS"]),
            bb=_to_optional_int_stat(row["BB"]),
            so=_to_optional_int_stat(row["SO"]),
            hbp=_to_optional_int_stat(row["HBP"]),
            sf=_to_optional_int_stat(row["SF"]),
            sh=_to_optional_int_stat(row["SH"]),
            gdp=_to_optional_int_stat(row["GDP"]),
            ibb=_to_optional_int_stat(row["IBB"]),
            avg=_to_optional_float(row["AVG"]),
            obp=_to_optional_float(row["OBP"]),
            slg=_to_optional_float(row["SLG"]),
            ops=_to_optional_float(row["OPS"]),
            woba=_to_optional_float(row["wOBA"]),
            wrc_plus=_to_optional_float(row["wRC+"]),
            war=_to_optional_float(row["WAR"]),
        )

    return mapper


def make_fg_pitching_mapper(
    players: list[Player],
) -> Callable[[pd.Series], PitchingStats | None]:
    fg_lookup = _build_fg_lookup(players)

    def mapper(row: pd.Series) -> PitchingStats | None:
        player_id = _resolve_fg_id(fg_lookup, row)
        if player_id is None:
            return None

        return PitchingStats(
            player_id=player_id,
            season=int(row["Season"]),
            source="fangraphs",
            w=_to_optional_int_stat(row["W"]),
            l=_to_optional_int_stat(row["L"]),
            g=_to_optional_int_stat(row["G"]),
            gs=_to_optional_int_stat(row["GS"]),
            sv=_to_optional_int_stat(row["SV"]),
            hld=_to_optional_int_stat(row.get("HLD")),
            h=_to_optional_int_stat(row["H"]),
            er=_to_optional_int_stat(row["ER"]),
            hr=_to_optional_int_stat(row["HR"]),
            bb=_to_optional_int_stat(row["BB"]),
            so=_to_optional_int_stat(row["SO"]),
            era=_to_optional_float(row["ERA"]),
            ip=_to_optional_float(row["IP"]),
            whip=_to_optional_float(row["WHIP"]),
            k_per_9=_to_optional_float(row["K/9"]),
            bb_per_9=_to_optional_float(row["BB/9"]),
            fip=_to_optional_float(row["FIP"]),
            xfip=_to_optional_float(row["xFIP"]),
            war=_to_optional_float(row["WAR"]),
        )

    return mapper


def _build_mlbam_lookup(players: list[Player]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for p in players:
        if p.mlbam_id is not None and p.id is not None:
            lookup[p.mlbam_id] = p.id
    return lookup


def _resolve_mlbam_id(mlbam_lookup: dict[int, int], row: pd.Series) -> int | None:
    mlb_id = row["mlbID"]
    if isinstance(mlb_id, float) and math.isnan(mlb_id):
        return None
    return mlbam_lookup.get(int(mlb_id))


def make_bref_batting_mapper(players: list[Player], *, season: int) -> Callable[[pd.Series], BattingStats | None]:
    mlbam_lookup = _build_mlbam_lookup(players)

    def mapper(row: pd.Series) -> BattingStats | None:
        player_id = _resolve_mlbam_id(mlbam_lookup, row)
        if player_id is None:
            return None

        return BattingStats(
            player_id=player_id,
            season=season,
            source="bbref",
            pa=_to_optional_int_stat(row["PA"]),
            ab=_to_optional_int_stat(row["AB"]),
            h=_to_optional_int_stat(row["H"]),
            doubles=_to_optional_int_stat(row["2B"]),
            triples=_to_optional_int_stat(row["3B"]),
            hr=_to_optional_int_stat(row["HR"]),
            rbi=_to_optional_int_stat(row["RBI"]),
            r=_to_optional_int_stat(row["R"]),
            sb=_to_optional_int_stat(row["SB"]),
            cs=_to_optional_int_stat(row["CS"]),
            bb=_to_optional_int_stat(row["BB"]),
            so=_to_optional_int_stat(row["SO"]),
            hbp=_to_optional_int_stat(row["HBP"]),
            sf=_to_optional_int_stat(row["SF"]),
            sh=_to_optional_int_stat(row["SH"]),
            gdp=_to_optional_int_stat(row["GDP"]),
            ibb=_to_optional_int_stat(row["IBB"]),
            avg=_to_optional_float(row["BA"]),
            obp=_to_optional_float(row["OBP"]),
            slg=_to_optional_float(row["SLG"]),
            ops=_to_optional_float(row["OPS"]),
        )

    return mapper


def make_bref_pitching_mapper(players: list[Player], *, season: int) -> Callable[[pd.Series], PitchingStats | None]:
    mlbam_lookup = _build_mlbam_lookup(players)

    def mapper(row: pd.Series) -> PitchingStats | None:
        player_id = _resolve_mlbam_id(mlbam_lookup, row)
        if player_id is None:
            return None

        return PitchingStats(
            player_id=player_id,
            season=season,
            source="bbref",
            w=_to_optional_int_stat(row["W"]),
            l=_to_optional_int_stat(row["L"]),
            g=_to_optional_int_stat(row["G"]),
            gs=_to_optional_int_stat(row["GS"]),
            sv=_to_optional_int_stat(row["SV"]),
            h=_to_optional_int_stat(row["H"]),
            er=_to_optional_int_stat(row["ER"]),
            hr=_to_optional_int_stat(row["HR"]),
            bb=_to_optional_int_stat(row["BB"]),
            so=_to_optional_int_stat(row["SO"]),
            era=_to_optional_float(row["ERA"]),
            ip=_to_optional_float(row["IP"]),
            whip=_to_optional_float(row["WHIP"]),
            k_per_9=_to_optional_float(row["SO9"]),
        )

    return mapper


def chadwick_row_to_player(row: pd.Series) -> Player | None:
    mlbam_id = _to_optional_int(row["key_mlbam"])
    if mlbam_id is None:
        return None

    return Player(
        name_first=str(row["name_first"]),
        name_last=str(row["name_last"]),
        mlbam_id=mlbam_id,
        fangraphs_id=_to_optional_int(row["key_fangraphs"]),
        bbref_id=_to_optional_str(row["key_bbref"]),
        retro_id=_to_optional_str(row["key_retro"]),
    )


def _to_required_int(value: Any) -> int | None:
    """Convert to int, returning None if NaN (signals the row should be skipped)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def statcast_pitch_mapper(row: pd.Series) -> StatcastPitch | None:
    game_pk = _to_required_int(row.get("game_pk"))
    batter_id = _to_required_int(row.get("batter"))
    pitcher_id = _to_required_int(row.get("pitcher"))
    at_bat_number = _to_required_int(row.get("at_bat_number"))
    pitch_number = _to_required_int(row.get("pitch_number"))

    if any(v is None for v in (game_pk, batter_id, pitcher_id, at_bat_number, pitch_number)):
        return None

    game_date = row.get("game_date")
    if game_date is None or (isinstance(game_date, float) and math.isnan(game_date)):
        return None

    return StatcastPitch(
        game_pk=game_pk,  # type: ignore[arg-type]
        game_date=str(game_date),
        batter_id=batter_id,  # type: ignore[arg-type]
        pitcher_id=pitcher_id,  # type: ignore[arg-type]
        at_bat_number=at_bat_number,  # type: ignore[arg-type]
        pitch_number=pitch_number,  # type: ignore[arg-type]
        pitch_type=_to_optional_str(row.get("pitch_type")),
        release_speed=_to_optional_float(row.get("release_speed")),
        release_spin_rate=_to_optional_float(row.get("release_spin_rate")),
        pfx_x=_to_optional_float(row.get("pfx_x")),
        pfx_z=_to_optional_float(row.get("pfx_z")),
        plate_x=_to_optional_float(row.get("plate_x")),
        plate_z=_to_optional_float(row.get("plate_z")),
        zone=_to_optional_int_stat(row.get("zone")),
        events=_to_optional_str(row.get("events")),
        description=_to_optional_str(row.get("description")),
        launch_speed=_to_optional_float(row.get("launch_speed")),
        launch_angle=_to_optional_float(row.get("launch_angle")),
        hit_distance_sc=_to_optional_float(row.get("hit_distance_sc")),
        barrel=_to_optional_int_stat(row.get("barrel")),
        estimated_ba_using_speedangle=_to_optional_float(row.get("estimated_ba_using_speedangle")),
        estimated_woba_using_speedangle=_to_optional_float(row.get("estimated_woba_using_speedangle")),
    )


def _collect_stats(row: pd.Series, column_map: dict[str, str]) -> dict[str, Any]:
    """Extract stats from a row, mapping CSV columns to canonical names, skipping NaN values."""
    stats: dict[str, Any] = {}
    for csv_col, stat_name in column_map.items():
        val = row.get(csv_col)
        if val is None:
            continue
        if isinstance(val, float) and math.isnan(val):
            continue
        stats[stat_name] = val
    return stats


_FG_BATTING_PROJECTION_COLUMNS: dict[str, str] = {
    "PA": "pa",
    "AB": "ab",
    "H": "h",
    "2B": "doubles",
    "3B": "triples",
    "HR": "hr",
    "RBI": "rbi",
    "R": "r",
    "SB": "sb",
    "CS": "cs",
    "BB": "bb",
    "SO": "so",
    "HBP": "hbp",
    "SF": "sf",
    "SH": "sh",
    "GDP": "gdp",
    "IBB": "ibb",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "wOBA": "woba",
    "wRC+": "wrc_plus",
    "WAR": "war",
}

_FG_PITCHING_PROJECTION_COLUMNS: dict[str, str] = {
    "W": "w",
    "L": "l",
    "G": "g",
    "GS": "gs",
    "SV": "sv",
    "HLD": "hld",
    "H": "h",
    "ER": "er",
    "HR": "hr",
    "BB": "bb",
    "SO": "so",
    "ERA": "era",
    "IP": "ip",
    "WHIP": "whip",
    "K/9": "k_per_9",
    "BB/9": "bb_per_9",
    "FIP": "fip",
    "xFIP": "xfip",
    "WAR": "war",
}


def _resolve_fg_projection_id(fg_lookup: dict[int, int], row: pd.Series) -> int | None:
    fg_id = row.get("playerid")
    if fg_id is None:
        return None
    if isinstance(fg_id, float) and math.isnan(fg_id):
        return None
    return fg_lookup.get(int(fg_id))


def make_fg_projection_batting_mapper(
    players: list[Player],
    *,
    season: int,
    system: str,
    version: str,
    source_type: str = "first_party",
) -> Callable[[pd.Series], Projection | None]:
    fg_lookup = _build_fg_lookup(players)

    def mapper(row: pd.Series) -> Projection | None:
        player_id = _resolve_fg_projection_id(fg_lookup, row)
        if player_id is None:
            return None

        stat_json = _collect_stats(row, _FG_BATTING_PROJECTION_COLUMNS)
        return Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type="batter",
            stat_json=stat_json,
            source_type=source_type,
        )

    return mapper


def make_fg_projection_pitching_mapper(
    players: list[Player],
    *,
    season: int,
    system: str,
    version: str,
    source_type: str = "first_party",
) -> Callable[[pd.Series], Projection | None]:
    fg_lookup = _build_fg_lookup(players)

    def mapper(row: pd.Series) -> Projection | None:
        player_id = _resolve_fg_projection_id(fg_lookup, row)
        if player_id is None:
            return None

        stat_json = _collect_stats(row, _FG_PITCHING_PROJECTION_COLUMNS)
        return Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type="pitcher",
            stat_json=stat_json,
            source_type=source_type,
        )

    return mapper
