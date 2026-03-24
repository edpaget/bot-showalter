from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.services.player_resolver import resolve_player


def _make_player(player_id: int, name: str, position: str = "OF") -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type=PlayerType.BATTER,
        position=position,
        value=10.0,
        category_z_scores={},
    )


POOL = [
    _make_player(1, "Mike Trout"),
    _make_player(2, "Shohei Ohtani"),
    _make_player(3, "Mookie Betts"),
    _make_player(4, "Mike Yastrzemski"),
    _make_player(5, "Ronald Acuna Jr."),
]


class TestExactMatch:
    def test_exact_full_name(self) -> None:
        result = resolve_player("Mike Trout", POOL)
        assert len(result) == 1
        assert result[0].player_id == 1

    def test_exact_case_insensitive(self) -> None:
        result = resolve_player("mike trout", POOL)
        assert len(result) == 1
        assert result[0].player_id == 1

    def test_exact_mixed_case(self) -> None:
        result = resolve_player("MIKE TROUT", POOL)
        assert len(result) == 1
        assert result[0].player_id == 1


class TestSubstringMatch:
    def test_last_name_substring(self) -> None:
        result = resolve_player("Trout", POOL)
        assert len(result) == 1
        assert result[0].player_id == 1

    def test_first_name_shared_returns_multiple(self) -> None:
        result = resolve_player("Mike", POOL)
        assert len(result) == 2
        ids = {r.player_id for r in result}
        assert ids == {1, 4}

    def test_substring_case_insensitive(self) -> None:
        result = resolve_player("ohtani", POOL)
        assert len(result) == 1
        assert result[0].player_id == 2

    def test_partial_last_name(self) -> None:
        result = resolve_player("Bett", POOL)
        assert len(result) == 1
        assert result[0].player_id == 3


class TestFuzzyMatch:
    def test_misspelling(self) -> None:
        result = resolve_player("Mike Trour", POOL)
        assert len(result) >= 1
        assert any(r.player_id == 1 for r in result)

    def test_close_spelling(self) -> None:
        result = resolve_player("Shohei Ohtni", POOL)
        assert len(result) >= 1
        assert any(r.player_id == 2 for r in result)


class TestNoMatch:
    def test_no_match_returns_empty(self) -> None:
        result = resolve_player("Zzzzzz Nonexistent", POOL)
        assert result == []


class TestEmptyPool:
    def test_empty_pool(self) -> None:
        result = resolve_player("Mike Trout", [])
        assert result == []
