from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import httpx

if TYPE_CHECKING:
    import pytest

from fantasy_baseball_manager.domain import YahooLeague, YahooTeam
from fantasy_baseball_manager.services.renewal_chain import walk_renewal_chain


def _make_league(
    league_key: str,
    season: int,
    game_key: str,
    renew: str | None = None,
) -> YahooLeague:
    return YahooLeague(
        league_key=league_key,
        name=f"League {season}",
        season=season,
        num_teams=10,
        draft_type="live",
        is_keeper=False,
        game_key=game_key,
        renew=renew,
    )


def _make_team(league_key: str, idx: int) -> YahooTeam:
    return YahooTeam(
        team_key=f"{league_key}.t.{idx}",
        league_key=league_key,
        team_id=idx,
        name=f"Team {idx}",
        manager_name=f"Manager {idx}",
        is_owned_by_user=idx == 1,
    )


class FakeLeagueRepo:
    """In-memory league repo for testing."""

    def __init__(self) -> None:
        self._leagues: dict[str, YahooLeague] = {}

    def upsert(self, league: YahooLeague) -> int:
        self._leagues[league.league_key] = league
        return 1

    def get_by_league_key(self, league_key: str) -> YahooLeague | None:
        return self._leagues.get(league_key)

    def get_all(self) -> list[YahooLeague]:
        return list(self._leagues.values())

    def seed(self, league: YahooLeague) -> None:
        self._leagues[league.league_key] = league


class FakeTeamRepo:
    """In-memory team repo for testing."""

    def __init__(self) -> None:
        self._teams: dict[str, list[YahooTeam]] = {}

    def upsert(self, team: YahooTeam) -> int:
        self._teams.setdefault(team.league_key, []).append(team)
        return 1

    def get_by_league_key(self, league_key: str) -> list[YahooTeam]:
        return self._teams.get(league_key, [])

    def get_user_team(self, league_key: str) -> YahooTeam | None:
        for t in self._teams.get(league_key, []):
            if t.is_owned_by_user:
                return t
        return None


class FakeLeagueSource:
    """Returns pre-configured leagues for given league keys."""

    def __init__(self, leagues: dict[str, tuple[YahooLeague, list[YahooTeam]]]) -> None:
        self._leagues = leagues

    def fetch(self, *, league_key: str, game_key: str) -> tuple[YahooLeague, list[YahooTeam]]:
        return self._leagues[league_key]


def _build_chain_fixtures() -> tuple[FakeLeagueSource, FakeLeagueRepo, FakeTeamRepo]:
    """Build a 3-season chain: 2025->2024->2023."""
    league_2025 = _make_league("449.l.100", 2025, "449", renew="422_200")
    league_2024 = _make_league("422.l.200", 2024, "422", renew="412_300")
    league_2023 = _make_league("412.l.300", 2023, "412", renew=None)

    teams_2025 = [_make_team("449.l.100", i) for i in range(1, 3)]
    teams_2024 = [_make_team("422.l.200", i) for i in range(1, 3)]
    teams_2023 = [_make_team("412.l.300", i) for i in range(1, 3)]

    source = FakeLeagueSource(
        {
            "449.l.100": (league_2025, teams_2025),
            "422.l.200": (league_2024, teams_2024),
            "412.l.300": (league_2023, teams_2023),
        }
    )

    league_repo = FakeLeagueRepo()
    team_repo = FakeTeamRepo()

    return source, league_repo, team_repo


class TestWalkRenewalChain:
    def test_three_season_chain(self) -> None:
        source, league_repo, team_repo = _build_chain_fixtures()

        result = walk_renewal_chain(
            league_source=source,
            league_repo=league_repo,
            team_repo=team_repo,
            league_key="449.l.100",
        )

        assert result == [
            ("449.l.100", 2025),
            ("422.l.200", 2024),
            ("412.l.300", 2023),
        ]

    def test_chain_breaks_when_renew_is_none(self) -> None:
        """Chain with only one league (no renew) returns single entry."""
        league = _make_league("449.l.100", 2025, "449", renew=None)
        source = FakeLeagueSource(
            {
                "449.l.100": (league, [_make_team("449.l.100", 1)]),
            }
        )
        league_repo = FakeLeagueRepo()
        team_repo = FakeTeamRepo()

        result = walk_renewal_chain(
            league_source=source,
            league_repo=league_repo,
            team_repo=team_repo,
            league_key="449.l.100",
        )

        assert result == [("449.l.100", 2025)]

    def test_chain_breaks_on_http_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Chain stops and logs warning when API call fails."""
        league_2025 = _make_league("449.l.100", 2025, "449", renew="422_200")

        # Source raises HTTPStatusError for the prior league
        def failing_fetch(*, league_key: str, game_key: str) -> Any:
            if league_key == "422.l.200":
                response = httpx.Response(status_code=404, request=httpx.Request("GET", "http://test"))
                raise httpx.HTTPStatusError("Not Found", request=response.request, response=response)
            return (league_2025, [_make_team("449.l.100", 1)])

        source = MagicMock()
        source.fetch = failing_fetch

        league_repo = FakeLeagueRepo()
        team_repo = FakeTeamRepo()

        with caplog.at_level(logging.WARNING):
            result = walk_renewal_chain(
                league_source=source,
                league_repo=league_repo,
                team_repo=team_repo,
                league_key="449.l.100",
            )

        assert result == [("449.l.100", 2025)]
        assert any("422.l.200" in msg for msg in caplog.messages)

    def test_since_parameter_limits_walk(self) -> None:
        source, league_repo, team_repo = _build_chain_fixtures()

        result = walk_renewal_chain(
            league_source=source,
            league_repo=league_repo,
            team_repo=team_repo,
            league_key="449.l.100",
            since=2024,
        )

        assert result == [
            ("449.l.100", 2025),
            ("422.l.200", 2024),
        ]

    def test_already_synced_league_not_refetched(self) -> None:
        """If the league is already in the repo, don't call the source."""
        source, league_repo, team_repo = _build_chain_fixtures()

        # Pre-seed the starting league
        league_2025 = _make_league("449.l.100", 2025, "449", renew="422_200")
        league_repo.seed(league_2025)

        # Use a tracking wrapper source
        call_keys: list[str] = []

        class TrackingSource:
            def fetch(self, *, league_key: str, game_key: str) -> tuple[YahooLeague, list[YahooTeam]]:
                call_keys.append(league_key)
                return source.fetch(league_key=league_key, game_key=game_key)

        walk_renewal_chain(
            league_source=TrackingSource(),
            league_repo=league_repo,
            team_repo=team_repo,
            league_key="449.l.100",
        )

        # 449.l.100 was already in the repo, so only 422.l.200 and 412.l.300 are fetched
        assert "449.l.100" not in call_keys
        assert "422.l.200" in call_keys
        assert "412.l.300" in call_keys

    def test_empty_renew_string_treated_as_none(self) -> None:
        """Yahoo sometimes returns '' instead of None for leagues with no prior season."""
        league = _make_league("449.l.100", 2025, "449", renew="")
        source = FakeLeagueSource(
            {
                "449.l.100": (league, [_make_team("449.l.100", 1)]),
            }
        )
        league_repo = FakeLeagueRepo()
        team_repo = FakeTeamRepo()

        result = walk_renewal_chain(
            league_source=source,
            league_repo=league_repo,
            team_repo=team_repo,
            league_key="449.l.100",
        )

        assert result == [("449.l.100", 2025)]
