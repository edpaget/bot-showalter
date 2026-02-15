import pandas as pd

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.ingest.column_maps import make_il_stint_mapper


def _make_players(*, mlbam_id: int = 545361, player_id: int = 1) -> list[Player]:
    return [Player(id=player_id, name_first="Mike", name_last="Trout", mlbam_id=mlbam_id)]


def _make_row(**overrides) -> pd.Series:
    defaults = {
        "transaction_id": 1,
        "mlbam_id": 545361,
        "date": "2024-05-15T00:00:00",
        "effective_date": "2024-05-15T00:00:00",
        "description": "Los Angeles Angels placed CF Mike Trout on the 15-day injured list. Left knee surgery.",
    }
    return pd.Series({**defaults, **overrides})


class TestMakeILStintMapper:
    def test_resolves_known_player(self) -> None:
        players = _make_players()
        mapper = make_il_stint_mapper(players, season=2024)

        result = mapper(_make_row())

        assert result is not None
        assert result.player_id == 1
        assert result.season == 2024
        assert result.start_date == "2024-05-15"
        assert result.il_type == "15"
        assert result.injury_location == "Left knee surgery"
        assert result.transaction_type == "placement"

    def test_returns_none_for_unknown_mlbam_id(self) -> None:
        players = _make_players()
        mapper = make_il_stint_mapper(players, season=2024)

        result = mapper(_make_row(mlbam_id=999999))

        assert result is None

    def test_returns_none_for_non_il_description(self) -> None:
        players = _make_players()
        mapper = make_il_stint_mapper(players, season=2024)

        result = mapper(_make_row(description="Los Angeles Angels placed CF Mike Trout on the paternity list."))

        assert result is None

    def test_extracts_activation(self) -> None:
        players = _make_players()
        mapper = make_il_stint_mapper(players, season=2024)

        result = mapper(
            _make_row(
                description="Los Angeles Angels activated CF Mike Trout from the 10-day injured list.",
            )
        )

        assert result is not None
        assert result.transaction_type == "activation"
        assert result.il_type == "10"
        assert result.injury_location is None

    def test_uses_effective_date_for_start_date(self) -> None:
        players = _make_players()
        mapper = make_il_stint_mapper(players, season=2024)

        result = mapper(
            _make_row(
                date="2024-06-01T00:00:00",
                effective_date="2024-05-29T00:00:00",
                description="Los Angeles Angels placed CF Mike Trout on the 15-day injured list"
                " retroactive to May 29, 2024. Meniscus tear.",
            )
        )

        assert result is not None
        assert result.start_date == "2024-05-29"

    def test_date_only_format(self) -> None:
        players = _make_players()
        mapper = make_il_stint_mapper(players, season=2024)

        result = mapper(_make_row(effective_date="2024-05-15"))

        assert result is not None
        assert result.start_date == "2024-05-15"
