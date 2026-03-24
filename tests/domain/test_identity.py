from enum import StrEnum

import pytest

from fantasy_baseball_manager.domain.identity import PlayerIdentity, PlayerType


class TestPlayerType:
    def test_is_str_enum(self) -> None:
        assert issubclass(PlayerType, StrEnum)

    def test_batter_value(self) -> None:
        assert PlayerType.BATTER == "batter"
        assert PlayerType.BATTER.value == "batter"

    def test_pitcher_value(self) -> None:
        assert PlayerType.PITCHER == "pitcher"
        assert PlayerType.PITCHER.value == "pitcher"

    def test_exactly_two_members(self) -> None:
        assert set(PlayerType) == {PlayerType.BATTER, PlayerType.PITCHER}

    def test_construct_from_string(self) -> None:
        assert PlayerType("batter") is PlayerType.BATTER
        assert PlayerType("pitcher") is PlayerType.PITCHER

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            PlayerType("invalid")


class TestPlayerIdentity:
    def test_fields(self) -> None:
        identity = PlayerIdentity(player_id=123, player_type=PlayerType.BATTER)
        assert identity.player_id == 123
        assert identity.player_type is PlayerType.BATTER

    def test_tuple_destructuring(self) -> None:
        identity = PlayerIdentity(123, PlayerType.PITCHER)
        pid, ptype = identity
        assert pid == 123
        assert ptype is PlayerType.PITCHER

    def test_tuple_indexing(self) -> None:
        identity = PlayerIdentity(42, PlayerType.BATTER)
        assert identity[0] == 42
        assert identity[1] is PlayerType.BATTER

    def test_hashable_and_dict_key(self) -> None:
        a = PlayerIdentity(1, PlayerType.BATTER)
        b = PlayerIdentity(1, PlayerType.BATTER)
        d = {a: "value"}
        assert d[b] == "value"

    def test_equality(self) -> None:
        a = PlayerIdentity(1, PlayerType.BATTER)
        b = PlayerIdentity(1, PlayerType.BATTER)
        c = PlayerIdentity(1, PlayerType.PITCHER)
        assert a == b
        assert a != c

    def test_tuple_compatible(self) -> None:
        identity = PlayerIdentity(1, PlayerType.BATTER)
        assert isinstance(identity, tuple)
        assert len(identity) == 2
