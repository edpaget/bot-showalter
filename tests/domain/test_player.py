import pytest

from fantasy_baseball_manager.domain.player import Player, Team


class TestPlayer:
    def test_construct_with_required_fields(self) -> None:
        player = Player(name_first="Mike", name_last="Trout")
        assert player.name_first == "Mike"
        assert player.name_last == "Trout"

    def test_optional_fields_default_to_none(self) -> None:
        player = Player(name_first="Mike", name_last="Trout")
        assert player.id is None
        assert player.mlbam_id is None
        assert player.fangraphs_id is None
        assert player.bbref_id is None
        assert player.retro_id is None
        assert player.bats is None
        assert player.throws is None
        assert player.birth_date is None

    def test_construct_with_all_fields(self) -> None:
        player = Player(
            id=1,
            name_first="Mike",
            name_last="Trout",
            mlbam_id=545361,
            fangraphs_id=10155,
            bbref_id="troutmi01",
            retro_id="troum001",
            bats="R",
            throws="R",
            birth_date="1991-08-07",
        )
        assert player.id == 1
        assert player.mlbam_id == 545361
        assert player.bbref_id == "troutmi01"

    def test_frozen(self) -> None:
        player = Player(name_first="Mike", name_last="Trout")
        with pytest.raises(AttributeError):
            player.name_first = "Shohei"  # type: ignore[misc]


class TestTeam:
    def test_construct_with_required_fields(self) -> None:
        team = Team(abbreviation="LAD", name="Los Angeles Dodgers", league="NL", division="W")
        assert team.abbreviation == "LAD"
        assert team.name == "Los Angeles Dodgers"

    def test_optional_id(self) -> None:
        team = Team(abbreviation="LAD", name="Los Angeles Dodgers", league="NL", division="W")
        assert team.id is None

    def test_frozen(self) -> None:
        team = Team(abbreviation="LAD", name="Los Angeles Dodgers", league="NL", division="W")
        with pytest.raises(AttributeError):
            team.abbreviation = "NYY"  # type: ignore[misc]
