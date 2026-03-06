from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.name_utils import normalize_name, resolve_players, strip_name_decorations
from tests.fakes.repos import FakePlayerRepo


class TestNormalizeName:
    def test_plain_name(self) -> None:
        assert normalize_name("Mike Trout") == "mike trout"

    def test_strips_jr_suffix(self) -> None:
        assert normalize_name("Bobby Witt Jr.") == "bobby witt"

    def test_strips_roman_numeral_suffix(self) -> None:
        assert normalize_name("Ken Griffey II") == "ken griffey"

    def test_strips_batter_parenthetical(self) -> None:
        assert normalize_name("Shohei Ohtani (Batter)") == "shohei ohtani"

    def test_strips_pitcher_parenthetical(self) -> None:
        assert normalize_name("Shohei Ohtani (Pitcher)") == "shohei ohtani"

    def test_strips_accents(self) -> None:
        assert normalize_name("Ronald Acuña") == "ronald acuna"

    def test_merges_dotted_initials(self) -> None:
        assert normalize_name("J.T. Realmuto") == "jt realmuto"

    def test_merges_spaced_dotted_initials(self) -> None:
        assert normalize_name("J. T. Realmuto") == "jt realmuto"

    def test_applies_nickname_alias(self) -> None:
        assert normalize_name("Matthew Boyd") == "matt boyd"
        assert normalize_name("Michael King") == "mike king"

    def test_combined_suffix_and_accent(self) -> None:
        assert normalize_name("Ronald Acuña Jr.") == "ronald acuna"


def _player(name_first: str, name_last: str, pid: int = 1) -> Player:
    return Player(id=pid, name_first=name_first, name_last=name_last, mlbam_id=pid)


class TestResolvePlayers:
    def test_first_last_with_accent(self) -> None:
        repo = FakePlayerRepo([_player("Cristopher", "Sánchez")])
        result = resolve_players(repo, "Cristopher Sanchez")
        assert len(result) == 1

    def test_last_only_with_accent(self) -> None:
        repo = FakePlayerRepo([_player("Ronald", "Acuña")])
        result = resolve_players(repo, "Acuna")
        assert len(result) == 1

    def test_comma_format(self) -> None:
        repo = FakePlayerRepo(
            [
                _player("Joe", "Smith", pid=1),
                _player("John", "Smith", pid=2),
            ]
        )
        result = resolve_players(repo, "Smith, Joe")
        assert len(result) == 1
        assert result[0].name_first == "Joe"

    def test_nickname_alias(self) -> None:
        repo = FakePlayerRepo([_player("Christopher", "Sale")])
        result = resolve_players(repo, "Chris Sale")
        assert len(result) == 1

    def test_single_word_last_name(self) -> None:
        repo = FakePlayerRepo([_player("Mike", "Trout")])
        result = resolve_players(repo, "Trout")
        assert len(result) == 1

    def test_no_match(self) -> None:
        repo = FakePlayerRepo([_player("Mike", "Trout")])
        result = resolve_players(repo, "Nobody")
        assert result == []

    def test_first_name_narrows_results(self) -> None:
        repo = FakePlayerRepo(
            [
                _player("Joe", "Smith", pid=1),
                _player("John", "Smith", pid=2),
            ]
        )
        result = resolve_players(repo, "Joe Smith")
        assert len(result) == 1
        assert result[0].name_first == "Joe"


class TestStripNameDecorations:
    def test_strips_jr_suffix(self) -> None:
        assert strip_name_decorations("Bobby Witt Jr.") == "Bobby Witt"

    def test_strips_parenthetical(self) -> None:
        assert strip_name_decorations("Shohei Ohtani (Batter)") == "Shohei Ohtani"

    def test_no_change_for_plain_name(self) -> None:
        assert strip_name_decorations("Mike Trout") == "Mike Trout"

    def test_strips_both(self) -> None:
        assert strip_name_decorations("Some Player Jr. (Batter)") == "Some Player"
