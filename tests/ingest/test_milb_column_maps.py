import math

import pandas as pd

from fantasy_baseball_manager.domain.minor_league_batting_stats import MinorLeagueBattingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.ingest.column_maps import make_milb_batting_mapper


def _make_player(*, player_id: int = 1, mlbam_id: int = 545361) -> Player:
    return Player(id=player_id, name_first="Mike", name_last="Trout", mlbam_id=mlbam_id)


def _make_row(**overrides: object) -> pd.Series:
    defaults: dict[str, object] = {
        "mlbam_id": 545361,
        "season": 2024,
        "level": "AAA",
        "league": "International League",
        "team": "Syracuse Mets",
        "g": 120,
        "pa": 500,
        "ab": 450,
        "h": 130,
        "doubles": 25,
        "triples": 3,
        "hr": 18,
        "r": 70,
        "rbi": 65,
        "bb": 40,
        "so": 100,
        "sb": 15,
        "cs": 5,
        "avg": 0.289,
        "obp": 0.350,
        "slg": 0.480,
        "age": 24.5,
        "hbp": 8,
        "sf": 4,
        "sh": 1,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


class TestMilbBattingMapper:
    def test_valid_row_maps_correctly(self) -> None:
        player = _make_player()
        mapper = make_milb_batting_mapper([player])

        result = mapper(_make_row())

        assert result is not None
        assert isinstance(result, MinorLeagueBattingStats)
        assert result.player_id == 1
        assert result.season == 2024
        assert result.level == "AAA"
        assert result.league == "International League"
        assert result.team == "Syracuse Mets"
        assert result.g == 120
        assert result.pa == 500
        assert result.ab == 450
        assert result.h == 130
        assert result.doubles == 25
        assert result.triples == 3
        assert result.hr == 18
        assert result.r == 70
        assert result.rbi == 65
        assert result.bb == 40
        assert result.so == 100
        assert result.sb == 15
        assert result.cs == 5
        assert result.hbp == 8
        assert result.sf == 4
        assert result.sh == 1

    def test_unknown_player_returns_none(self) -> None:
        player = _make_player(mlbam_id=999999)
        mapper = make_milb_batting_mapper([player])

        result = mapper(_make_row(mlbam_id=545361))

        assert result is None

    def test_nan_mlbam_id_returns_none(self) -> None:
        player = _make_player()
        mapper = make_milb_batting_mapper([player])

        result = mapper(_make_row(mlbam_id=float("nan")))

        assert result is None

    def test_optional_fields_handle_nan(self) -> None:
        player = _make_player()
        mapper = make_milb_batting_mapper([player])

        result = mapper(_make_row(hbp=math.nan, sf=math.nan, sh=math.nan))

        assert result is not None
        assert result.hbp is None
        assert result.sf is None
        assert result.sh is None
