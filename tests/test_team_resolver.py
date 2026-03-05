from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import Team
from fantasy_baseball_manager.team_resolver import TeamResolver

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import TeamRepo


class FakeTeamRepo:
    """In-memory TeamRepo for testing."""

    def __init__(self, teams: list[Team]) -> None:
        self._teams = teams
        self._next_id = 1

    def upsert(self, team: Team) -> int:
        self._teams.append(team)
        self._next_id += 1
        return self._next_id - 1

    def get_by_abbreviation(self, abbreviation: str) -> Team | None:
        for t in self._teams:
            if t.abbreviation == abbreviation:
                return t
        return None

    def all(self) -> list[Team]:
        return list(self._teams)


# Verify the fake satisfies the protocol
_: type[TeamRepo] = FakeTeamRepo

_TEAMS = [
    Team(abbreviation="NYY", name="New York Yankees", league="AL", division="E", id=1),
    Team(abbreviation="NYM", name="New York Mets", league="NL", division="E", id=2),
    Team(abbreviation="LAD", name="Los Angeles Dodgers", league="NL", division="W", id=3),
    Team(abbreviation="KC", name="Kansas City Royals", league="AL", division="C", id=4),
    Team(abbreviation="SF", name="San Francisco Giants", league="NL", division="W", id=5),
    Team(abbreviation="TB", name="Tampa Bay Rays", league="AL", division="E", id=6),
    Team(abbreviation="STL", name="St. Louis Cardinals", league="NL", division="C", id=7),
    Team(abbreviation="CHC", name="Chicago Cubs", league="NL", division="C", id=8),
    Team(abbreviation="CWS", name="Chicago White Sox", league="AL", division="C", id=9),
]


def _make_resolver() -> TeamResolver:
    return TeamResolver(FakeTeamRepo(_TEAMS[:]))


class TestExactAbbreviation:
    def test_modern_abbreviation(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("NYY") == ["NYY"]

    def test_abbreviation_case_insensitive(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("nyy") == ["NYY"]


class TestLahmanAbbreviation:
    def test_lahman_alias_kca(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("KCA") == ["KC"]

    def test_lahman_alias_lan(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("LAN") == ["LAD"]

    def test_lahman_alias_nya(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("NYA") == ["NYY"]


class TestFullName:
    def test_full_name(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("New York Yankees") == ["NYY"]

    def test_full_name_case_insensitive(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("new york yankees") == ["NYY"]


class TestCityOnly:
    def test_ambiguous_city(self) -> None:
        resolver = _make_resolver()
        result = resolver.resolve("New York")
        assert sorted(result) == ["NYM", "NYY"]

    def test_unambiguous_city(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("Kansas City") == ["KC"]

    def test_chicago_ambiguous(self) -> None:
        resolver = _make_resolver()
        result = resolver.resolve("Chicago")
        assert sorted(result) == ["CHC", "CWS"]


class TestNickname:
    def test_nickname(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("Yankees") == ["NYY"]

    def test_nickname_case_insensitive(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("dodgers") == ["LAD"]

    def test_nickname_mets(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("Mets") == ["NYM"]


class TestFuzzy:
    def test_misspelled_yankees(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("Yankeez") == ["NYY"]

    def test_misspelled_dodgers(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("Dogers") == ["LAD"]


class TestNoMatch:
    def test_no_match(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("xyzabc") == []

    def test_empty_query(self) -> None:
        resolver = _make_resolver()
        assert resolver.resolve("") == []
