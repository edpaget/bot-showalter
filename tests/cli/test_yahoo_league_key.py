"""Tests for resolve_league_key helper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.cli.commands.yahoo import resolve_league_key
from fantasy_baseball_manager.domain import YahooLeague

if TYPE_CHECKING:
    import pytest


@dataclass
class FakeClient:
    """Minimal fake Yahoo client that returns predefined game keys."""

    game_keys: dict[int, str]

    def get_game_key(self, season: int) -> str:
        return self.game_keys[season]

    # Satisfy the broader protocol with no-ops
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


@dataclass
class FakeLeagueRepo:
    """Minimal fake YahooLeagueRepo backed by a dict."""

    leagues: dict[str, YahooLeague]

    def get_by_league_key(self, league_key: str) -> YahooLeague | None:
        return self.leagues.get(league_key)

    def upsert(self, league: YahooLeague) -> int:
        return 0

    def get_all(self) -> list[YahooLeague]:
        return list(self.leagues.values())


@dataclass
class FakeCtx:
    """Minimal stand-in for YahooContext with only the fields resolve_league_key needs."""

    client: FakeClient
    yahoo_league_repo: FakeLeagueRepo


class TestResolveLeagueKey:
    def test_current_season_returns_direct_key(self) -> None:
        ctx = FakeCtx(
            client=FakeClient(game_keys={2026: "459"}),
            yahoo_league_repo=FakeLeagueRepo(leagues={}),
        )
        result = resolve_league_key(ctx, league_id=12345, season=2026, current_year=2026)  # type: ignore[arg-type]
        assert result == "459.l.12345"

    def test_prior_season_follows_renew_chain(self) -> None:
        current_league = YahooLeague(
            league_key="459.l.12345",
            name="Test League",
            season=2026,
            num_teams=12,
            draft_type="live",
            is_keeper=True,
            game_key="459",
            renew="458_99999",
        )
        ctx = FakeCtx(
            client=FakeClient(game_keys={2025: "458", 2026: "459"}),
            yahoo_league_repo=FakeLeagueRepo(leagues={"459.l.12345": current_league}),
        )
        result = resolve_league_key(ctx, league_id=12345, season=2025, current_year=2026)  # type: ignore[arg-type]
        assert result == "458.l.99999"

    def test_prior_season_no_stored_league_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = FakeCtx(
            client=FakeClient(game_keys={2025: "458", 2026: "459"}),
            yahoo_league_repo=FakeLeagueRepo(leagues={}),
        )
        with caplog.at_level(logging.WARNING):
            result = resolve_league_key(ctx, league_id=12345, season=2025, current_year=2026)  # type: ignore[arg-type]
        assert result == "458.l.12345"
        assert "No renew chain" in caplog.text

    def test_prior_season_stored_league_no_renew_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        current_league = YahooLeague(
            league_key="459.l.12345",
            name="Test League",
            season=2026,
            num_teams=12,
            draft_type="live",
            is_keeper=False,
            game_key="459",
            renew=None,
        )
        ctx = FakeCtx(
            client=FakeClient(game_keys={2025: "458", 2026: "459"}),
            yahoo_league_repo=FakeLeagueRepo(leagues={"459.l.12345": current_league}),
        )
        with caplog.at_level(logging.WARNING):
            result = resolve_league_key(ctx, league_id=12345, season=2025, current_year=2026)  # type: ignore[arg-type]
        assert result == "458.l.12345"
        assert "No renew chain" in caplog.text

    def test_future_season_returns_direct_key(self) -> None:
        ctx = FakeCtx(
            client=FakeClient(game_keys={2027: "460"}),
            yahoo_league_repo=FakeLeagueRepo(leagues={}),
        )
        result = resolve_league_key(ctx, league_id=12345, season=2027, current_year=2026)  # type: ignore[arg-type]
        assert result == "460.l.12345"
