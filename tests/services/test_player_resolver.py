from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.player_alias import PlayerAlias
from fantasy_baseball_manager.services.player_name_resolver import PlayerNameResolver
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


class _FakeAliasRepo:
    """In-memory alias repo for testing."""

    def __init__(self, aliases: list[PlayerAlias] | None = None) -> None:
        self._aliases: list[PlayerAlias] = list(aliases) if aliases else []

    def upsert(self, alias: PlayerAlias) -> int:
        self._aliases.append(alias)
        return len(self._aliases)

    def upsert_batch(self, aliases: list[PlayerAlias]) -> int:
        self._aliases.extend(aliases)
        return len(aliases)

    def find_by_name(self, alias_name: str) -> list[PlayerAlias]:
        return [a for a in self._aliases if a.alias_name == alias_name]

    def find_by_player(self, player_id: int) -> list[PlayerAlias]:
        return [a for a in self._aliases if a.player_id == player_id]

    def delete_by_player(self, player_id: int) -> int:
        before = len(self._aliases)
        self._aliases = [a for a in self._aliases if a.player_id != player_id]
        return before - len(self._aliases)


class TestResolverFirst:
    def test_resolver_matches_by_alias(self) -> None:
        alias_repo = _FakeAliasRepo(
            [
                PlayerAlias(
                    alias_name="big mike",
                    player_id=1,
                    player_type=PlayerType.BATTER,
                    source="manual",
                ),
            ]
        )
        resolver = PlayerNameResolver(alias_repo)

        # "Big Mike" won't match any pool name via substring/fuzzy, but resolver knows it
        result = resolve_player("Big Mike", POOL, resolver=resolver)
        assert len(result) == 1
        assert result[0].player_id == 1

    def test_falls_through_when_resolver_returns_nothing(self) -> None:
        alias_repo = _FakeAliasRepo()
        resolver = PlayerNameResolver(alias_repo)

        # No aliases registered — should fall through to substring match
        result = resolve_player("Trout", POOL, resolver=resolver)
        assert len(result) == 1
        assert result[0].player_id == 1

    def test_falls_through_when_resolver_match_not_in_pool(self) -> None:
        alias_repo = _FakeAliasRepo(
            [
                PlayerAlias(
                    alias_name="ghost player",
                    player_id=9999,
                    player_type=PlayerType.BATTER,
                    source="manual",
                ),
            ]
        )
        resolver = PlayerNameResolver(alias_repo)

        # Resolver matches id=9999 but nobody in pool has that id
        result = resolve_player("Ghost Player", POOL, resolver=resolver)
        assert result == []
