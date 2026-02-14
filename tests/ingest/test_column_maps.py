import pandas as pd

from fantasy_baseball_manager.ingest.column_maps import chadwick_row_to_player


def _make_row(
    *,
    name_first: str = "Mike",
    name_last: str = "Trout",
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
        assert player.position is None
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
