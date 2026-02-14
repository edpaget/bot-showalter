import pandas as pd

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.ingest.column_maps import (
    make_bref_batting_mapper,
    make_bref_pitching_mapper,
    make_fg_batting_mapper,
    make_fg_pitching_mapper,
)


def _trout() -> Player:
    return Player(
        name_first="Mike",
        name_last="Trout",
        id=1,
        mlbam_id=545361,
        fangraphs_id=10155,
    )


def _ohtani() -> Player:
    return Player(
        name_first="Shohei",
        name_last="Ohtani",
        id=2,
        mlbam_id=660271,
        fangraphs_id=19755,
    )


def _fg_batting_row(
    *,
    idfg: int | str = 10155,
    season: int = 2024,
    pa: int | float = 600,
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
    hbp: int = 5,
    sf: int = 4,
    sh: int = 1,
    gdp: int = 10,
    ibb: int = 8,
    avg: float = 0.302,
    obp: float = 0.395,
    slg: float = 0.575,
    ops: float = 0.970,
    woba: float = 0.410,
    wrc_plus: float = 170.0,
    war: float = 8.5,
) -> pd.Series:
    return pd.Series(
        {
            "IDfg": idfg,
            "Season": season,
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
            "HBP": hbp,
            "SF": sf,
            "SH": sh,
            "GDP": gdp,
            "IBB": ibb,
            "AVG": avg,
            "OBP": obp,
            "SLG": slg,
            "OPS": ops,
            "wOBA": woba,
            "wRC+": wrc_plus,
            "WAR": war,
        }
    )


class TestFgBattingMapper:
    def test_complete_row(self) -> None:
        mapper = make_fg_batting_mapper([_trout()])
        result = mapper(_fg_batting_row())

        assert result is not None
        assert result.player_id == 1
        assert result.season == 2024
        assert result.source == "fangraphs"
        assert result.pa == 600
        assert result.ab == 530
        assert result.h == 160
        assert result.doubles == 30
        assert result.triples == 5
        assert result.hr == 35
        assert result.rbi == 90
        assert result.r == 100
        assert result.sb == 15
        assert result.cs == 3
        assert result.bb == 60
        assert result.so == 120
        assert result.hbp == 5
        assert result.sf == 4
        assert result.sh == 1
        assert result.gdp == 10
        assert result.ibb == 8
        assert result.avg == 0.302
        assert result.obp == 0.395
        assert result.slg == 0.575
        assert result.ops == 0.970
        assert result.woba == 0.410
        assert result.wrc_plus == 170.0
        assert result.war == 8.5

    def test_unknown_id_returns_none(self) -> None:
        mapper = make_fg_batting_mapper([_trout()])
        result = mapper(_fg_batting_row(idfg=99999))
        assert result is None

    def test_nan_int_becomes_none(self) -> None:
        mapper = make_fg_batting_mapper([_trout()])
        result = mapper(_fg_batting_row(pa=float("nan")))
        assert result is not None
        assert result.pa is None

    def test_nan_float_becomes_none(self) -> None:
        mapper = make_fg_batting_mapper([_trout()])
        result = mapper(_fg_batting_row(avg=float("nan")))
        assert result is not None
        assert result.avg is None

    def test_second_player(self) -> None:
        mapper = make_fg_batting_mapper([_trout(), _ohtani()])
        result = mapper(_fg_batting_row(idfg=19755))
        assert result is not None
        assert result.player_id == 2


def _fg_pitching_row(
    *,
    idfg: int | str = 10155,
    season: int = 2024,
    w: int = 15,
    losses: int = 8,
    g: int = 32,
    gs: int = 32,
    sv: int = 0,
    hld: int | None = 0,
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
    xfip: float = 3.25,
    war: float = 6.0,
    include_hld: bool = True,
) -> pd.Series:
    data: dict = {
        "IDfg": idfg,
        "Season": season,
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
        "xFIP": xfip,
        "WAR": war,
    }
    if include_hld:
        data["HLD"] = hld
    return pd.Series(data)


class TestFgPitchingMapper:
    def test_complete_row(self) -> None:
        mapper = make_fg_pitching_mapper([_trout()])
        result = mapper(_fg_pitching_row())

        assert result is not None
        assert result.player_id == 1
        assert result.season == 2024
        assert result.source == "fangraphs"
        assert result.w == 15
        assert result.l == 8
        assert result.g == 32
        assert result.gs == 32
        assert result.sv == 0
        assert result.hld == 0
        assert result.h == 140
        assert result.er == 55
        assert result.hr == 18
        assert result.bb == 45
        assert result.so == 220
        assert result.era == 2.85
        assert result.ip == 190.1
        assert result.whip == 0.97
        assert result.k_per_9 == 10.4
        assert result.bb_per_9 == 2.1
        assert result.fip == 3.10
        assert result.xfip == 3.25
        assert result.war == 6.0

    def test_unknown_id_returns_none(self) -> None:
        mapper = make_fg_pitching_mapper([_trout()])
        result = mapper(_fg_pitching_row(idfg=99999))
        assert result is None

    def test_missing_hld_column_becomes_none(self) -> None:
        mapper = make_fg_pitching_mapper([_trout()])
        result = mapper(_fg_pitching_row(include_hld=False))
        assert result is not None
        assert result.hld is None

    def test_nan_float_becomes_none(self) -> None:
        mapper = make_fg_pitching_mapper([_trout()])
        result = mapper(_fg_pitching_row(era=float("nan")))
        assert result is not None
        assert result.era is None


def _bref_batting_row(
    *,
    mlb_id: int | float = 545361,
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
    hbp: int = 5,
    sf: int = 4,
    sh: int = 1,
    gdp: int = 10,
    ibb: int = 8,
    ba: float = 0.302,
    obp: float = 0.395,
    slg: float = 0.575,
    ops: float = 0.970,
) -> pd.Series:
    return pd.Series(
        {
            "mlbID": mlb_id,
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
            "HBP": hbp,
            "SF": sf,
            "SH": sh,
            "GDP": gdp,
            "IBB": ibb,
            "BA": ba,
            "OBP": obp,
            "SLG": slg,
            "OPS": ops,
        }
    )


class TestBrefBattingMapper:
    def test_complete_row(self) -> None:
        mapper = make_bref_batting_mapper([_trout()], season=2024)
        result = mapper(_bref_batting_row())

        assert result is not None
        assert result.player_id == 1
        assert result.season == 2024
        assert result.source == "bbref"
        assert result.pa == 600
        assert result.ab == 530
        assert result.h == 160
        assert result.doubles == 30
        assert result.triples == 5
        assert result.hr == 35
        assert result.rbi == 90
        assert result.r == 100
        assert result.sb == 15
        assert result.cs == 3
        assert result.bb == 60
        assert result.so == 120
        assert result.hbp == 5
        assert result.sf == 4
        assert result.sh == 1
        assert result.gdp == 10
        assert result.ibb == 8
        assert result.avg == 0.302
        assert result.obp == 0.395
        assert result.slg == 0.575
        assert result.ops == 0.970

    def test_season_from_factory(self) -> None:
        mapper = make_bref_batting_mapper([_trout()], season=2023)
        result = mapper(_bref_batting_row())
        assert result is not None
        assert result.season == 2023

    def test_unknown_mlbam_id_returns_none(self) -> None:
        mapper = make_bref_batting_mapper([_trout()], season=2024)
        result = mapper(_bref_batting_row(mlb_id=999999))
        assert result is None

    def test_nan_mlbam_id_returns_none(self) -> None:
        mapper = make_bref_batting_mapper([_trout()], season=2024)
        result = mapper(_bref_batting_row(mlb_id=float("nan")))
        assert result is None

    def test_advanced_stats_are_none(self) -> None:
        mapper = make_bref_batting_mapper([_trout()], season=2024)
        result = mapper(_bref_batting_row())
        assert result is not None
        assert result.woba is None
        assert result.wrc_plus is None
        assert result.war is None


def _bref_pitching_row(
    *,
    mlb_id: int | float = 545361,
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
    so9: float = 10.4,
) -> pd.Series:
    return pd.Series(
        {
            "mlbID": mlb_id,
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
            "SO9": so9,
        }
    )


class TestBrefPitchingMapper:
    def test_complete_row(self) -> None:
        mapper = make_bref_pitching_mapper([_trout()], season=2024)
        result = mapper(_bref_pitching_row())

        assert result is not None
        assert result.player_id == 1
        assert result.season == 2024
        assert result.source == "bbref"
        assert result.w == 15
        assert result.l == 8
        assert result.g == 32
        assert result.gs == 32
        assert result.sv == 0
        assert result.h == 140
        assert result.er == 55
        assert result.hr == 18
        assert result.bb == 45
        assert result.so == 220
        assert result.era == 2.85
        assert result.ip == 190.1
        assert result.whip == 0.97
        assert result.k_per_9 == 10.4

    def test_season_from_factory(self) -> None:
        mapper = make_bref_pitching_mapper([_trout()], season=2023)
        result = mapper(_bref_pitching_row())
        assert result is not None
        assert result.season == 2023

    def test_unknown_id_returns_none(self) -> None:
        mapper = make_bref_pitching_mapper([_trout()], season=2024)
        result = mapper(_bref_pitching_row(mlb_id=999999))
        assert result is None

    def test_missing_stats_are_none(self) -> None:
        mapper = make_bref_pitching_mapper([_trout()], season=2024)
        result = mapper(_bref_pitching_row())
        assert result is not None
        assert result.hld is None
        assert result.bb_per_9 is None
        assert result.fip is None
        assert result.xfip is None
        assert result.war is None
