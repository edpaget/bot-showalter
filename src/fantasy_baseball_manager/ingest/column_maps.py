import math
from collections.abc import Callable
from typing import Any

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.il_stint import ILStint
from fantasy_baseball_manager.domain.minor_league_batting_stats import MinorLeagueBattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.player import Player, Team
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection, StatDistribution
from fantasy_baseball_manager.domain.roster_stint import RosterStint
from fantasy_baseball_manager.domain.sprint_speed import SprintSpeed
from fantasy_baseball_manager.domain.statcast_pitch import StatcastPitch
from fantasy_baseball_manager.ingest.il_parser import parse_il_transaction


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


def _resolve_fg_id(fg_lookup: dict[int, int], row: dict[str, Any]) -> int | None:
    fg_id = row["IDfg"]
    if isinstance(fg_id, float) and math.isnan(fg_id):
        return None
    return fg_lookup.get(int(fg_id))


def make_fg_batting_mapper(
    players: list[Player],
) -> Callable[[dict[str, Any]], BattingStats | None]:
    fg_lookup = _build_fg_lookup(players)

    def mapper(row: dict[str, Any]) -> BattingStats | None:
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
) -> Callable[[dict[str, Any]], PitchingStats | None]:
    fg_lookup = _build_fg_lookup(players)

    def mapper(row: dict[str, Any]) -> PitchingStats | None:
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


def chadwick_row_to_player(row: dict[str, Any]) -> Player | None:
    mlbam_id = _to_optional_int(row["key_mlbam"])
    if mlbam_id is None:
        return None

    first = _to_optional_str(row["name_first"])
    last = _to_optional_str(row["name_last"])
    if first is None and last is None:
        return None

    return Player(
        name_first=first or "",
        name_last=last or "",
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


def statcast_pitch_mapper(row: dict[str, Any]) -> StatcastPitch | None:
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
        estimated_slg_using_speedangle=_to_optional_float(row.get("estimated_slg_using_speedangle")),
        hc_x=_to_optional_float(row.get("hc_x")),
        hc_y=_to_optional_float(row.get("hc_y")),
        stand=_to_optional_str(row.get("stand")),
        release_extension=_to_optional_float(row.get("release_extension")),
    )


def _build_retro_lookup(players: list[Player]) -> dict[str, Player]:
    lookup: dict[str, Player] = {}
    for p in players:
        if p.retro_id is not None:
            lookup[p.retro_id] = p
    return lookup


def _build_birth_date(year: int | None, month: int | None, day: int | None) -> str | None:
    if year is None:
        return None
    m = month if month is not None else 1
    d = day if day is not None else 1
    return f"{year:04d}-{m:02d}-{d:02d}"


def make_lahman_bio_mapper(
    players: list[Player],
) -> Callable[[dict[str, Any]], Player | None]:
    retro_lookup = _build_retro_lookup(players)

    def mapper(row: dict[str, Any]) -> Player | None:
        retro_id = _to_optional_str(row["retroID"])
        if retro_id is None:
            return None
        player = retro_lookup.get(retro_id)
        if player is None:
            return None

        year = _to_optional_int_stat(row.get("birthYear"))
        month = _to_optional_int_stat(row.get("birthMonth"))
        day = _to_optional_int_stat(row.get("birthDay"))
        birth_date = _build_birth_date(year, month, day)

        return Player(
            id=player.id,
            name_first=player.name_first,
            name_last=player.name_last,
            mlbam_id=player.mlbam_id,
            fangraphs_id=player.fangraphs_id,
            bbref_id=player.bbref_id,
            retro_id=player.retro_id,
            bats=_to_optional_str(row.get("bats")),
            throws=_to_optional_str(row.get("throws")),
            birth_date=birth_date,
        )

    return mapper


def _collect_stats(row: dict[str, Any], column_map: dict[str, str]) -> dict[str, Any]:
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


_REQUIRED_PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)


def extract_distributions(
    row: dict[str, Any],
    column_map: dict[str, str],
) -> list[StatDistribution]:
    distributions: list[StatDistribution] = []
    for csv_col, stat_name in column_map.items():
        pct_keys = [f"{csv_col}_p{p}" for p in _REQUIRED_PERCENTILES]
        pct_vals: list[float] = []
        for key in pct_keys:
            val = row.get(key)
            if val is None:
                break
            if isinstance(val, float) and math.isnan(val):
                break
            pct_vals.append(float(val))
        if len(pct_vals) != len(_REQUIRED_PERCENTILES):
            continue
        mean = _to_optional_float(row.get(f"{csv_col}_mean"))
        std = _to_optional_float(row.get(f"{csv_col}_std"))
        distributions.append(
            StatDistribution(
                stat=stat_name,
                p10=pct_vals[0],
                p25=pct_vals[1],
                p50=pct_vals[2],
                p75=pct_vals[3],
                p90=pct_vals[4],
                mean=mean,
                std=std,
            )
        )
    return distributions


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
    "1B": "singles",
    "G": "g",
    "ISO": "iso",
    "BABIP": "babip",
    "BB%": "bb_pct",
    "K%": "k_pct",
    "wRC": "wrc",
    "wRAA": "wraa",
    "BsR": "bsr",
    "Off": "off",
    "Def": "def_",
    "Fld": "fld",
    "Spd": "spd",
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
    "QS": "qs",
    "BS": "bs",
    "TBF": "tbf",
    "R": "r",
    "HBP": "hbp",
    "IBB": "ibb",
    "K/BB": "k_per_bb",
    "HR/9": "hr_per_9",
    "K%": "k_pct",
    "BB%": "bb_pct",
    "BABIP": "babip",
    "LOB%": "lob_pct",
    "GB%": "gb_pct",
    "HR/FB": "hr_per_fb",
    "RA9-WAR": "ra9_war",
    "AVG": "avg",
}


def _resolve_fg_projection_id(
    fg_lookup: dict[int, int],
    mlbam_lookup: dict[int, int],
    row: dict[str, Any],
) -> int | None:
    player_id: int | None = None
    fg_id = row.get("PlayerId")
    if fg_id is not None and not (isinstance(fg_id, float) and math.isnan(fg_id)):
        fg_str = str(fg_id)
        if not fg_str.startswith("sa"):
            try:
                player_id = fg_lookup.get(int(float(fg_str)))
            except ValueError, OverflowError:
                pass
    if player_id is not None:
        return player_id
    mlbam_id = row.get("MLBAMID")
    if mlbam_id is None:
        return None
    if isinstance(mlbam_id, float) and math.isnan(mlbam_id):
        return None
    try:
        return mlbam_lookup.get(int(mlbam_id))
    except ValueError, OverflowError:
        return None


def make_fg_projection_batting_mapper(
    players: list[Player],
    *,
    season: int,
    system: str,
    version: str,
    source_type: str = "first_party",
) -> Callable[[dict[str, Any]], Projection | None]:
    fg_lookup = _build_fg_lookup(players)
    mlbam_lookup = _build_mlbam_lookup(players)

    def mapper(row: dict[str, Any]) -> Projection | None:
        player_id = _resolve_fg_projection_id(fg_lookup, mlbam_lookup, row)
        if player_id is None:
            return None

        stat_json = _collect_stats(row, _FG_BATTING_PROJECTION_COLUMNS)
        dists = extract_distributions(row, _FG_BATTING_PROJECTION_COLUMNS)
        distributions = {d.stat: d for d in dists} if dists else None
        return Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type="batter",
            stat_json=stat_json,
            source_type=source_type,
            distributions=distributions,
        )

    return mapper


def make_fg_projection_pitching_mapper(
    players: list[Player],
    *,
    season: int,
    system: str,
    version: str,
    source_type: str = "first_party",
) -> Callable[[dict[str, Any]], Projection | None]:
    fg_lookup = _build_fg_lookup(players)
    mlbam_lookup = _build_mlbam_lookup(players)

    def mapper(row: dict[str, Any]) -> Projection | None:
        player_id = _resolve_fg_projection_id(fg_lookup, mlbam_lookup, row)
        if player_id is None:
            return None

        stat_json = _collect_stats(row, _FG_PITCHING_PROJECTION_COLUMNS)
        dists = extract_distributions(row, _FG_PITCHING_PROJECTION_COLUMNS)
        distributions = {d.stat: d for d in dists} if dists else None
        return Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type="pitcher",
            stat_json=stat_json,
            source_type=source_type,
            distributions=distributions,
        )

    return mapper


def _extract_date(value: Any) -> str:
    """Extract YYYY-MM-DD from a datetime string or date-only string."""
    s = str(value)
    return s[:10]


def make_il_stint_mapper(
    players: list[Player],
    *,
    season: int,
) -> Callable[[dict[str, Any]], ILStint | None]:
    mlbam_lookup = _build_mlbam_lookup(players)

    def mapper(row: dict[str, Any]) -> ILStint | None:
        mlb_id = row.get("mlbam_id")
        if mlb_id is None:
            return None
        if isinstance(mlb_id, float) and math.isnan(mlb_id):
            return None
        player_id = mlbam_lookup.get(int(mlb_id))
        if player_id is None:
            return None

        description = row.get("description")
        if description is None:
            return None
        parsed = parse_il_transaction(str(description))
        if parsed is None:
            return None

        effective_date = row.get("effective_date")
        if effective_date is None:
            return None
        start_date = _extract_date(effective_date)

        return ILStint(
            player_id=player_id,
            season=season,
            start_date=start_date,
            il_type=parsed.il_type,
            injury_location=parsed.injury_description,
            transaction_type=parsed.transaction_type,
        )

    return mapper


def _build_bbref_lookup(players: list[Player]) -> dict[str, Player]:
    lookup: dict[str, Player] = {}
    for p in players:
        if p.bbref_id is not None:
            lookup[p.bbref_id] = p
    return lookup


def _build_team_abbrev_lookup(teams: list[Team]) -> dict[str, int]:
    return {t.abbreviation: t.id for t in teams if t.id is not None}


def make_position_appearance_mapper(
    players: list[Player],
) -> Callable[[dict[str, Any]], PositionAppearance | None]:
    bbref_lookup = _build_bbref_lookup(players)

    def mapper(row: dict[str, Any]) -> PositionAppearance | None:
        bbref_id = _to_optional_str(row.get("playerID"))
        if bbref_id is None:
            return None
        player = bbref_lookup.get(bbref_id)
        if player is None or player.id is None:
            return None
        return PositionAppearance(
            player_id=player.id,
            season=int(row["yearID"]),
            position=str(row["position"]),
            games=int(row["games"]),
        )

    return mapper


def make_roster_stint_mapper(
    players: list[Player],
    teams: list[Team],
) -> Callable[[dict[str, Any]], RosterStint | None]:
    bbref_lookup = _build_bbref_lookup(players)
    team_lookup = _build_team_abbrev_lookup(teams)

    def mapper(row: dict[str, Any]) -> RosterStint | None:
        bbref_id = _to_optional_str(row.get("playerID"))
        if bbref_id is None:
            return None
        player = bbref_lookup.get(bbref_id)
        if player is None or player.id is None:
            return None
        team_abbrev = _to_optional_str(row.get("teamID"))
        if team_abbrev is None:
            return None
        team_id = team_lookup.get(team_abbrev)
        if team_id is None:
            return None
        season = int(row["yearID"])
        return RosterStint(
            player_id=player.id,
            team_id=team_id,
            season=season,
            start_date=f"{season}-03-01",
        )

    return mapper


def lahman_team_row_to_team(row: dict[str, Any]) -> Team | None:
    abbrev = _to_optional_str(row.get("teamID"))
    name = _to_optional_str(row.get("name"))
    if abbrev is None or name is None:
        return None
    league = _to_optional_str(row.get("lgID")) or ""
    division = _to_optional_str(row.get("divID")) or ""
    return Team(abbreviation=abbrev, name=name, league=league, division=division)


def make_milb_batting_mapper(
    players: list[Player],
) -> Callable[[dict[str, Any]], MinorLeagueBattingStats | None]:
    mlbam_lookup = _build_mlbam_lookup(players)

    def mapper(row: dict[str, Any]) -> MinorLeagueBattingStats | None:
        raw_id = row["mlbam_id"]
        if isinstance(raw_id, float) and math.isnan(raw_id):
            return None
        player_id = mlbam_lookup.get(int(raw_id))
        if player_id is None:
            return None

        return MinorLeagueBattingStats(
            player_id=player_id,
            season=int(row["season"]),
            level=str(row["level"]),
            league=str(row["league"]),
            team=str(row["team"]),
            g=int(row["g"]),
            pa=int(row["pa"]),
            ab=int(row["ab"]),
            h=int(row["h"]),
            doubles=int(row["doubles"]),
            triples=int(row["triples"]),
            hr=int(row["hr"]),
            r=int(row["r"]),
            rbi=int(row["rbi"]),
            bb=int(row["bb"]),
            so=int(row["so"]),
            sb=int(row["sb"]),
            cs=int(row["cs"]),
            avg=float(row["avg"]),
            obp=float(row["obp"]),
            slg=float(row["slg"]),
            age=float(row["age"]),
            hbp=_to_optional_int_stat(row.get("hbp")),
            sf=_to_optional_int_stat(row.get("sf")),
            sh=_to_optional_int_stat(row.get("sh")),
        )

    return mapper


def make_sprint_speed_mapper(
    *,
    season: int,
) -> Callable[[dict[str, Any]], SprintSpeed | None]:
    def mapper(row: dict[str, Any]) -> SprintSpeed | None:
        raw_id = row.get("player_id")
        if raw_id is None:
            return None
        if isinstance(raw_id, float) and math.isnan(raw_id):
            return None
        return SprintSpeed(
            mlbam_id=int(raw_id),
            season=season,
            sprint_speed=_to_optional_float(row.get("sprint_speed")),
            hp_to_1b=_to_optional_float(row.get("hp_to_1b")),
            bolts=_to_optional_int_stat(row.get("bolts")),
            competitive_runs=_to_optional_int_stat(row.get("competitive_runs")),
        )

    return mapper
