from fantasy_baseball_manager.pipeline.stages.identity_enricher import (
    PlayerIdentityEnricher,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates
from fantasy_baseball_manager.player_id.mapper import SfbbMapper


def _make_mapper() -> SfbbMapper:
    return SfbbMapper(
        yahoo_to_fg={},
        fg_to_yahoo={},
        fg_to_mlbam={"fg1": "mlb1", "fg2": "mlb2"},
        mlbam_to_fg={"mlb1": "fg1", "mlb2": "fg2"},
    )


def _make_player_rates(player_id: str = "fg1", name: str = "Test Player") -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        rates={"hr": 0.04},
        opportunities=600.0,
    )


class TestPlayerIdentityEnricher:
    def test_stamps_player_with_mlbam_id(self) -> None:
        enricher = PlayerIdentityEnricher(mapper=_make_mapper())
        players = [_make_player_rates("fg1", "Player One")]

        result = enricher.adjust(players)

        assert len(result) == 1
        assert result[0].player is not None
        assert result[0].player.fangraphs_id == "fg1"
        assert result[0].player.mlbam_id == "mlb1"
        assert result[0].player.name == "Player One"

    def test_preserves_rates_and_metadata(self) -> None:
        enricher = PlayerIdentityEnricher(mapper=_make_mapper())
        players = [_make_player_rates()]

        result = enricher.adjust(players)

        assert result[0].rates == {"hr": 0.04}
        assert result[0].opportunities == 600.0
        assert result[0].year == 2025
        assert result[0].age == 28

    def test_handles_unmapped_player(self) -> None:
        enricher = PlayerIdentityEnricher(mapper=_make_mapper())
        players = [_make_player_rates("unknown_fg", "Unknown")]

        result = enricher.adjust(players)

        assert len(result) == 1
        assert result[0].player is not None
        assert result[0].player.fangraphs_id == "unknown_fg"
        assert result[0].player.mlbam_id is None

    def test_enriches_multiple_players(self) -> None:
        enricher = PlayerIdentityEnricher(mapper=_make_mapper())
        players = [
            _make_player_rates("fg1", "Player One"),
            _make_player_rates("fg2", "Player Two"),
            _make_player_rates("fg_unknown", "Player Three"),
        ]

        result = enricher.adjust(players)

        assert len(result) == 3
        assert result[0].player is not None
        assert result[0].player.mlbam_id == "mlb1"
        assert result[1].player is not None
        assert result[1].player.mlbam_id == "mlb2"
        assert result[2].player is not None
        assert result[2].player.mlbam_id is None

    def test_empty_list(self) -> None:
        enricher = PlayerIdentityEnricher(mapper=_make_mapper())

        result = enricher.adjust([])

        assert result == []
