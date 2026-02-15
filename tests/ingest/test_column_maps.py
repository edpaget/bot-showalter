import pandas as pd

from fantasy_baseball_manager.domain.player import Player, Team
from fantasy_baseball_manager.domain.projection import StatDistribution
from fantasy_baseball_manager.ingest.column_maps import (
    _to_optional_float,
    chadwick_row_to_player,
    extract_distributions,
    lahman_team_row_to_team,
    make_lahman_bio_mapper,
    make_position_appearance_mapper,
    make_roster_stint_mapper,
)


def _make_row(
    *,
    name_first: str | float = "Mike",
    name_last: str | float = "Trout",
    key_mlbam: int | float = 545361,
    key_fangraphs: int | float = 10155,
    key_bbref: str | float = "troutmi01",
    key_retro: str | float = "troum001",
) -> pd.Series:
    return pd.Series(
        {
            "name_first": name_first,
            "name_last": name_last,
            "key_mlbam": key_mlbam,
            "key_fangraphs": key_fangraphs,
            "key_bbref": key_bbref,
            "key_retro": key_retro,
            "mlb_played_first": 2011.0,
            "mlb_played_last": 2024.0,
        }
    )


class TestChadwickRowToPlayer:
    def test_complete_row(self) -> None:
        player = chadwick_row_to_player(_make_row())
        assert player is not None
        assert player.name_first == "Mike"
        assert player.name_last == "Trout"
        assert player.mlbam_id == 545361
        assert player.fangraphs_id == 10155
        assert player.bbref_id == "troutmi01"
        assert player.retro_id == "troum001"

    def test_missing_mlbam_returns_none(self) -> None:
        row = _make_row(key_mlbam=float("nan"))
        assert chadwick_row_to_player(row) is None

    def test_sentinel_mlbam_returns_none(self) -> None:
        row = _make_row(key_mlbam=-1)
        assert chadwick_row_to_player(row) is None

    def test_nan_fangraphs_becomes_none(self) -> None:
        row = _make_row(key_fangraphs=float("nan"))
        player = chadwick_row_to_player(row)
        assert player is not None
        assert player.fangraphs_id is None

    def test_sentinel_fangraphs_becomes_none(self) -> None:
        row = _make_row(key_fangraphs=-1)
        player = chadwick_row_to_player(row)
        assert player is not None
        assert player.fangraphs_id is None

    def test_nan_bbref_becomes_none(self) -> None:
        row = _make_row(key_bbref=float("nan"))
        player = chadwick_row_to_player(row)
        assert player is not None
        assert player.bbref_id is None

    def test_nan_retro_becomes_none(self) -> None:
        row = _make_row(key_retro=float("nan"))
        player = chadwick_row_to_player(row)
        assert player is not None
        assert player.retro_id is None

    def test_empty_string_bbref_becomes_none(self) -> None:
        row = _make_row(key_bbref="")
        player = chadwick_row_to_player(row)
        assert player is not None
        assert player.bbref_id is None

    def test_fields_not_in_chadwick_are_none(self) -> None:
        player = chadwick_row_to_player(_make_row())
        assert player is not None
        assert player.bats is None
        assert player.throws is None
        assert player.birth_date is None
        assert player.id is None

    def test_mlbam_id_is_int_not_float(self) -> None:
        row = _make_row(key_mlbam=545361.0)
        player = chadwick_row_to_player(row)
        assert player is not None
        assert isinstance(player.mlbam_id, int)

    def test_fangraphs_id_is_int_not_float(self) -> None:
        row = _make_row(key_fangraphs=10155.0)
        player = chadwick_row_to_player(row)
        assert player is not None
        assert isinstance(player.fangraphs_id, int)

    def test_nan_name_first_gives_empty_string(self) -> None:
        row = _make_row(name_first=float("nan"))
        result = chadwick_row_to_player(row)
        assert result is not None
        assert result.name_first == ""

    def test_nan_name_last_gives_empty_string(self) -> None:
        row = _make_row(name_last=float("nan"))
        result = chadwick_row_to_player(row)
        assert result is not None
        assert result.name_last == ""

    def test_both_names_nan_returns_none(self) -> None:
        row = _make_row(name_first=float("nan"), name_last=float("nan"))
        assert chadwick_row_to_player(row) is None


def _make_lahman_row(
    *,
    retroID: str | float = "troum001",
    birthYear: int | float = 1991,
    birthMonth: int | float = 8,
    birthDay: int | float = 7,
    bats: str | float = "R",
    throws: str | float = "R",
) -> pd.Series:
    return pd.Series(
        {
            "retroID": retroID,
            "bbrefID": "troutmi01",
            "birthYear": birthYear,
            "birthMonth": birthMonth,
            "birthDay": birthDay,
            "bats": bats,
            "throws": throws,
        }
    )


_TROUT = Player(
    name_first="Mike",
    name_last="Trout",
    id=1,
    mlbam_id=545361,
    fangraphs_id=10155,
    bbref_id="troutmi01",
    retro_id="troum001",
)

_OHTANI = Player(
    name_first="Shohei",
    name_last="Ohtani",
    id=2,
    mlbam_id=660271,
    fangraphs_id=19755,
    bbref_id="ohtansh01",
    retro_id="ohtas001",
)


class TestMakeLahmanBioMapper:
    def test_enriches_matched_player(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        row = _make_lahman_row()
        result = mapper(row)
        assert result is not None
        assert result.mlbam_id == 545361
        assert result.birth_date == "1991-08-07"
        assert result.bats == "R"
        assert result.throws == "R"

    def test_preserves_existing_fields(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        result = mapper(_make_lahman_row())
        assert result is not None
        assert result.name_first == "Mike"
        assert result.name_last == "Trout"
        assert result.fangraphs_id == 10155
        assert result.bbref_id == "troutmi01"
        assert result.retro_id == "troum001"
        assert result.id == 1

    def test_unmatched_retro_id_returns_none(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        row = _make_lahman_row(retroID="xxxxx999")
        assert mapper(row) is None

    def test_nan_retro_id_returns_none(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        row = _make_lahman_row(retroID=float("nan"))
        assert mapper(row) is None

    def test_year_month_day_all_present(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        result = mapper(_make_lahman_row(birthYear=1991, birthMonth=8, birthDay=7))
        assert result is not None
        assert result.birth_date == "1991-08-07"

    def test_year_and_month_only(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        result = mapper(_make_lahman_row(birthDay=float("nan")))
        assert result is not None
        assert result.birth_date == "1991-08-01"

    def test_year_only(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        result = mapper(_make_lahman_row(birthMonth=float("nan"), birthDay=float("nan")))
        assert result is not None
        assert result.birth_date == "1991-01-01"

    def test_no_year_gives_none_birth_date(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        result = mapper(
            _make_lahman_row(
                birthYear=float("nan"),
                birthMonth=float("nan"),
                birthDay=float("nan"),
            )
        )
        assert result is not None
        assert result.birth_date is None

    def test_nan_bats_becomes_none(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        result = mapper(_make_lahman_row(bats=float("nan")))
        assert result is not None
        assert result.bats is None

    def test_nan_throws_becomes_none(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT])
        result = mapper(_make_lahman_row(throws=float("nan")))
        assert result is not None
        assert result.throws is None

    def test_multiple_players_lookup(self) -> None:
        mapper = make_lahman_bio_mapper([_TROUT, _OHTANI])
        row = pd.Series(
            {
                "retroID": "ohtas001",
                "bbrefID": "ohtansh01",
                "birthYear": 1994,
                "birthMonth": 7,
                "birthDay": 5,
                "bats": "L",
                "throws": "R",
            }
        )
        result = mapper(row)
        assert result is not None
        assert result.mlbam_id == 660271
        assert result.birth_date == "1994-07-05"
        assert result.bats == "L"


class TestExtractDistributions:
    _COLUMN_MAP: dict[str, str] = {"HR": "hr", "AVG": "avg"}

    def test_extracts_single_stat(self) -> None:
        row = pd.Series({"HR": 35, "HR_p10": 20.0, "HR_p25": 25.0, "HR_p50": 33.0, "HR_p75": 40.0, "HR_p90": 48.0})
        result = extract_distributions(row, {"HR": "hr"})
        assert len(result) == 1
        assert result[0] == StatDistribution(stat="hr", p10=20.0, p25=25.0, p50=33.0, p75=40.0, p90=48.0)

    def test_extracts_multiple_stats(self) -> None:
        row = pd.Series(
            {
                "HR": 35,
                "HR_p10": 20.0,
                "HR_p25": 25.0,
                "HR_p50": 33.0,
                "HR_p75": 40.0,
                "HR_p90": 48.0,
                "AVG": 0.300,
                "AVG_p10": 0.260,
                "AVG_p25": 0.275,
                "AVG_p50": 0.300,
                "AVG_p75": 0.320,
                "AVG_p90": 0.340,
            }
        )
        result = extract_distributions(row, self._COLUMN_MAP)
        assert len(result) == 2
        stats = {d.stat for d in result}
        assert stats == {"hr", "avg"}

    def test_no_percentile_columns_returns_empty(self) -> None:
        row = pd.Series({"HR": 35, "AVG": 0.300})
        result = extract_distributions(row, self._COLUMN_MAP)
        assert result == []

    def test_partial_percentiles_skipped(self) -> None:
        row = pd.Series({"HR": 35, "HR_p10": 20.0, "HR_p90": 48.0})
        result = extract_distributions(row, {"HR": "hr"})
        assert result == []

    def test_nan_percentile_skipped(self) -> None:
        row = pd.Series(
            {"HR": 35, "HR_p10": 20.0, "HR_p25": float("nan"), "HR_p50": 33.0, "HR_p75": 40.0, "HR_p90": 48.0}
        )
        result = extract_distributions(row, {"HR": "hr"})
        assert result == []

    def test_optional_mean_and_std(self) -> None:
        row = pd.Series(
            {
                "HR": 35,
                "HR_p10": 20.0,
                "HR_p25": 25.0,
                "HR_p50": 33.0,
                "HR_p75": 40.0,
                "HR_p90": 48.0,
                "HR_mean": 32.5,
                "HR_std": 8.2,
            }
        )
        result = extract_distributions(row, {"HR": "hr"})
        assert len(result) == 1
        assert result[0].mean == 32.5
        assert result[0].std == 8.2


class TestToOptionalFloat:
    def test_valid_float(self) -> None:
        assert _to_optional_float(3.14) == 3.14

    def test_int_to_float(self) -> None:
        result = _to_optional_float(5)
        assert result == 5.0
        assert isinstance(result, float)

    def test_none_returns_none(self) -> None:
        assert _to_optional_float(None) is None

    def test_nan_returns_none(self) -> None:
        assert _to_optional_float(float("nan")) is None

    def test_zero(self) -> None:
        result = _to_optional_float(0)
        assert result == 0.0
        assert isinstance(result, float)

    def test_string_float(self) -> None:
        assert _to_optional_float("3.14") == 3.14


def _make_appearance_row(
    *,
    playerID: str | float = "troutmi01",
    yearID: int = 2023,
    position: str = "CF",
    games: int = 120,
) -> pd.Series:
    return pd.Series({"playerID": playerID, "yearID": yearID, "teamID": "LAA", "position": position, "games": games})


class TestMakePositionAppearanceMapper:
    def test_matched_player_maps_correctly(self) -> None:
        mapper = make_position_appearance_mapper([_TROUT])
        result = mapper(_make_appearance_row())
        assert result is not None
        assert result.player_id == 1
        assert result.season == 2023
        assert result.position == "CF"
        assert result.games == 120

    def test_unmatched_player_returns_none(self) -> None:
        mapper = make_position_appearance_mapper([_TROUT])
        result = mapper(_make_appearance_row(playerID="xxxxx999"))
        assert result is None

    def test_nan_player_id_returns_none(self) -> None:
        mapper = make_position_appearance_mapper([_TROUT])
        result = mapper(_make_appearance_row(playerID=float("nan")))
        assert result is None

    def test_games_value_preserved(self) -> None:
        mapper = make_position_appearance_mapper([_TROUT])
        result = mapper(_make_appearance_row(games=55))
        assert result is not None
        assert result.games == 55


_TEST_TEAM = Team(abbreviation="LAA", name="Los Angeles Angels", league="AL", division="W", id=10)


def _make_roster_row(
    *,
    playerID: str | float = "troutmi01",
    yearID: int = 2023,
    teamID: str | float = "LAA",
) -> pd.Series:
    return pd.Series({"playerID": playerID, "yearID": yearID, "teamID": teamID})


class TestMakeRosterStintMapper:
    def test_matched_player_and_team_maps_correctly(self) -> None:
        mapper = make_roster_stint_mapper([_TROUT], [_TEST_TEAM])
        result = mapper(_make_roster_row())
        assert result is not None
        assert result.player_id == 1
        assert result.team_id == 10
        assert result.season == 2023
        assert result.start_date == "2023-03-01"

    def test_unmatched_player_returns_none(self) -> None:
        mapper = make_roster_stint_mapper([_TROUT], [_TEST_TEAM])
        result = mapper(_make_roster_row(playerID="xxxxx999"))
        assert result is None

    def test_unmatched_team_returns_none(self) -> None:
        mapper = make_roster_stint_mapper([_TROUT], [_TEST_TEAM])
        result = mapper(_make_roster_row(teamID="NYY"))
        assert result is None

    def test_start_date_format(self) -> None:
        mapper = make_roster_stint_mapper([_TROUT], [_TEST_TEAM])
        result = mapper(_make_roster_row(yearID=2020))
        assert result is not None
        assert result.start_date == "2020-03-01"

    def test_nan_player_id_returns_none(self) -> None:
        mapper = make_roster_stint_mapper([_TROUT], [_TEST_TEAM])
        result = mapper(_make_roster_row(playerID=float("nan")))
        assert result is None

    def test_nan_team_id_returns_none(self) -> None:
        mapper = make_roster_stint_mapper([_TROUT], [_TEST_TEAM])
        result = mapper(_make_roster_row(teamID=float("nan")))
        assert result is None


def _make_team_row(
    *,
    teamID: str | float = "LAA",
    name: str | float = "Los Angeles Angels",
    lgID: str | float = "AL",
    divID: str | float = "W",
) -> pd.Series:
    return pd.Series({"teamID": teamID, "name": name, "lgID": lgID, "divID": divID, "yearID": 2023})


class TestLahmanTeamRowToTeam:
    def test_valid_row(self) -> None:
        result = lahman_team_row_to_team(_make_team_row())
        assert result is not None
        assert result.abbreviation == "LAA"
        assert result.name == "Los Angeles Angels"
        assert result.league == "AL"
        assert result.division == "W"

    def test_missing_abbreviation_returns_none(self) -> None:
        result = lahman_team_row_to_team(_make_team_row(teamID=float("nan")))
        assert result is None

    def test_missing_name_returns_none(self) -> None:
        result = lahman_team_row_to_team(_make_team_row(name=float("nan")))
        assert result is None

    def test_missing_league_defaults_to_empty(self) -> None:
        result = lahman_team_row_to_team(_make_team_row(lgID=float("nan")))
        assert result is not None
        assert result.league == ""

    def test_missing_division_defaults_to_empty(self) -> None:
        result = lahman_team_row_to_team(_make_team_row(divID=float("nan")))
        assert result is not None
        assert result.division == ""
