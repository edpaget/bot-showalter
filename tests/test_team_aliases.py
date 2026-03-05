from fantasy_baseball_manager.team_aliases import (
    REVERSE_ALIASES,
    TEAM_ALIASES,
    to_lahman,
    to_modern,
)


class TestTeamAliases:
    def test_to_modern_known_alias(self) -> None:
        assert to_modern("NYA") == "NYY"
        assert to_modern("LAN") == "LAD"
        assert to_modern("CHA") == "CWS"

    def test_to_modern_passthrough(self) -> None:
        assert to_modern("NYY") == "NYY"
        assert to_modern("BOS") == "BOS"

    def test_to_lahman_known_alias(self) -> None:
        assert to_lahman("NYY") == "NYA"
        assert to_lahman("LAD") == "LAN"
        assert to_lahman("CWS") == "CHA"

    def test_to_lahman_passthrough(self) -> None:
        assert to_lahman("BOS") == "BOS"
        assert to_lahman("NYA") == "NYA"

    def test_reverse_aliases_covers_all(self) -> None:
        assert len(REVERSE_ALIASES) == len(TEAM_ALIASES)
        for lahman, modern in TEAM_ALIASES.items():
            assert REVERSE_ALIASES[modern] == lahman
