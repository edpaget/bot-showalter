import pandas as pd

from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.ingest.pybaseball_source import (
    BrefBattingSource,
    BrefPitchingSource,
    ChadwickSource,
    FgBattingSource,
    FgPitchingSource,
    LahmanPeopleSource,
    StatcastSource,
    _build_position_column,
    _translate_fg_params,
)


class TestChadwickSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(ChadwickSource(), DataSource)

    def test_source_type(self) -> None:
        assert ChadwickSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert ChadwickSource().source_detail == "chadwick_register"


class TestFgBattingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(FgBattingSource(), DataSource)

    def test_source_type(self) -> None:
        assert FgBattingSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert FgBattingSource().source_detail == "fg_batting_data"


class TestFgPitchingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(FgPitchingSource(), DataSource)

    def test_source_type(self) -> None:
        assert FgPitchingSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert FgPitchingSource().source_detail == "fg_pitching_data"


class TestBrefBattingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(BrefBattingSource(), DataSource)

    def test_source_type(self) -> None:
        assert BrefBattingSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert BrefBattingSource().source_detail == "batting_stats_bref"


class TestBrefPitchingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(BrefPitchingSource(), DataSource)

    def test_source_type(self) -> None:
        assert BrefPitchingSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert BrefPitchingSource().source_detail == "pitching_stats_bref"


class TestTranslateFgParams:
    def test_translates_season_to_start_end(self) -> None:
        result = _translate_fg_params({"season": 2023})
        assert result == {"start_season": 2023, "end_season": 2023, "qual": 0}

    def test_passthrough_when_no_season(self) -> None:
        result = _translate_fg_params({"start_season": 2022, "end_season": 2023})
        assert result == {"start_season": 2022, "end_season": 2023, "qual": 0}

    def test_does_not_override_explicit_start_end(self) -> None:
        result = _translate_fg_params({"season": 2023, "start_season": 2020})
        assert result == {"start_season": 2020, "end_season": 2023, "qual": 0}

    def test_defaults_qual_to_zero(self) -> None:
        result = _translate_fg_params({})
        assert result["qual"] == 0

    def test_does_not_override_explicit_qual(self) -> None:
        result = _translate_fg_params({"qual": 100})
        assert result["qual"] == 100


class TestLahmanPeopleSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(LahmanPeopleSource(), DataSource)

    def test_source_type(self) -> None:
        assert LahmanPeopleSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert LahmanPeopleSource().source_detail == "lahman_people"


class TestStatcastSource:
    def test_satisfies_datasource_protocol(self) -> None:
        assert isinstance(StatcastSource(), DataSource)

    def test_source_type(self) -> None:
        assert StatcastSource().source_type == "pybaseball"

    def test_source_detail(self) -> None:
        assert StatcastSource().source_detail == "statcast"


def _make_appearances(**overrides: object) -> dict[str, object]:
    """Build an Appearances row with all position columns defaulting to 0."""
    base: dict[str, object] = {
        "playerID": "troutmi01",
        "yearID": 2023,
        "G_p": 0,
        "G_c": 0,
        "G_1b": 0,
        "G_2b": 0,
        "G_3b": 0,
        "G_ss": 0,
        "G_lf": 0,
        "G_cf": 0,
        "G_rf": 0,
        "G_dh": 0,
    }
    base.update(overrides)
    return base


class TestBuildPositionColumn:
    def test_single_position_above_threshold(self) -> None:
        df = pd.DataFrame([_make_appearances(G_ss=500)])
        result = _build_position_column(df)
        row = result.set_index("playerID").loc["troutmi01"]
        assert row["eligible_positions"] == "SS"

    def test_multi_position_sorted_by_games(self) -> None:
        df = pd.DataFrame([_make_appearances(G_ss=500, G_2b=200)])
        result = _build_position_column(df)
        row = result.set_index("playerID").loc["troutmi01"]
        assert row["eligible_positions"] == "SS,2B"

    def test_pitcher_only(self) -> None:
        df = pd.DataFrame([_make_appearances(G_p=300)])
        result = _build_position_column(df)
        row = result.set_index("playerID").loc["troutmi01"]
        assert row["eligible_positions"] == "P"

    def test_below_threshold_excluded(self) -> None:
        df = pd.DataFrame([_make_appearances(G_p=5)])
        result = _build_position_column(df)
        row = result.set_index("playerID").loc["troutmi01"]
        assert pd.isna(row["eligible_positions"])

    def test_aggregates_across_seasons(self) -> None:
        rows = [
            _make_appearances(yearID=2022, G_cf=80),
            _make_appearances(yearID=2023, G_cf=90, G_rf=15),
        ]
        df = pd.DataFrame(rows)
        result = _build_position_column(df)
        row = result.set_index("playerID").loc["troutmi01"]
        assert row["eligible_positions"] == "CF,RF"

    def test_custom_min_games(self) -> None:
        df = pd.DataFrame([_make_appearances(G_1b=15)])
        result = _build_position_column(df, min_games=20)
        row = result.set_index("playerID").loc["troutmi01"]
        assert pd.isna(row["eligible_positions"])

    def test_two_way_player(self) -> None:
        df = pd.DataFrame([_make_appearances(G_dh=100, G_p=80)])
        result = _build_position_column(df)
        row = result.set_index("playerID").loc["troutmi01"]
        assert row["eligible_positions"] == "DH,P"

    def test_multiple_players(self) -> None:
        rows = [
            _make_appearances(playerID="troutmi01", G_cf=500),
            {**_make_appearances(playerID="ohtansh01", G_dh=200, G_p=150)},
        ]
        df = pd.DataFrame(rows)
        result = _build_position_column(df)
        indexed = result.set_index("playerID")
        assert indexed.loc["troutmi01"]["eligible_positions"] == "CF"
        assert indexed.loc["ohtansh01"]["eligible_positions"] == "DH,P"
