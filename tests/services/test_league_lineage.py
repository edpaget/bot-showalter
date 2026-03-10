from fantasy_baseball_manager.domain.yahoo_league import YahooLeague
from fantasy_baseball_manager.services.league_lineage import find_league_lineage


def _league(key: str, season: int, renew: str | None = None) -> YahooLeague:
    game_key = key.split(".l.")[0]
    return YahooLeague(
        league_key=key,
        name="Test League",
        season=season,
        num_teams=12,
        draft_type="live",
        is_keeper=False,
        game_key=game_key,
        renew=renew,
    )


class TestFindLeagueLineage:
    def test_chain_of_three(self) -> None:
        leagues = [
            _league("422.l.91300", 2025, renew="412_91300"),
            _league("412.l.91300", 2024, renew="403_91300"),
            _league("403.l.91300", 2023),
        ]
        result = find_league_lineage(leagues, "422.l.91300")
        assert result == ["403.l.91300", "412.l.91300", "422.l.91300"]

    def test_single_league_no_renew(self) -> None:
        leagues = [_league("422.l.91300", 2025)]
        result = find_league_lineage(leagues, "422.l.91300")
        assert result == ["422.l.91300"]

    def test_start_key_not_found(self) -> None:
        leagues = [_league("422.l.91300", 2025)]
        result = find_league_lineage(leagues, "999.l.00000")
        assert result == []

    def test_broken_chain(self) -> None:
        """Renew points to a league not in the list."""
        leagues = [_league("422.l.91300", 2025, renew="412_91300")]
        result = find_league_lineage(leagues, "422.l.91300")
        assert result == ["422.l.91300"]
