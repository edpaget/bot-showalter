from fantasy_baseball_manager.cli._live_server import create_live_draft_app
from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.valuation import Valuation


def _league() -> LeagueSettings:
    batting_cats = (CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),)
    pitching_cats = (CategoryConfig(key="w", name="W", stat_type=StatType.COUNTING, direction=Direction.HIGHER),)
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=14,
        roster_pitchers=9,
        batting_categories=batting_cats,
        pitching_categories=pitching_cats,
    )


def _valuations() -> list[Valuation]:
    return [
        Valuation(
            player_id=1,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026.1",
            player_type="batter",
            position="OF",
            value=30.0,
            rank=1,
            category_scores={"hr": 2.0},
        ),
        Valuation(
            player_id=2,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026.1",
            player_type="pitcher",
            position="SP",
            value=25.0,
            rank=2,
            category_scores={"w": 1.5},
        ),
        Valuation(
            player_id=3,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026.1",
            player_type="batter",
            position="1B",
            value=20.0,
            rank=3,
            category_scores={"hr": 1.0},
        ),
    ]


def _names() -> dict[int, str]:
    return {1: "Mike Trout", 2: "Gerrit Cole", 3: "Pete Alonso"}


def _adp() -> list[ADP]:
    return [
        ADP(player_id=1, season=2026, provider="fp", overall_pick=3.0, rank=3, positions="OF"),
        ADP(player_id=2, season=2026, provider="fp", overall_pick=10.0, rank=10, positions="SP"),
    ]


class TestLiveServerRoutes:
    def test_get_root_returns_html_with_players(self) -> None:
        app = create_live_draft_app(_valuations(), _league(), _names(), _adp())
        client = app.test_client()
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Mike Trout" in html
        assert "Gerrit Cole" in html
        assert "Pete Alonso" in html

    def test_get_root_has_auto_refresh(self) -> None:
        app = create_live_draft_app(_valuations(), _league(), _names(), _adp())
        client = app.test_client()
        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert 'http-equiv="refresh"' in html

    def test_post_draft_removes_player(self) -> None:
        app = create_live_draft_app(_valuations(), _league(), _names(), _adp())
        client = app.test_client()

        resp = client.post("/draft/1")
        assert resp.status_code == 200

        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert "Mike Trout" not in html
        assert "Gerrit Cole" in html
        assert "Pete Alonso" in html

    def test_post_draft_unknown_player_returns_404(self) -> None:
        app = create_live_draft_app(_valuations(), _league(), _names(), _adp())
        client = app.test_client()

        resp = client.post("/draft/999")
        assert resp.status_code == 404

    def test_post_reset_restores_pool(self) -> None:
        app = create_live_draft_app(_valuations(), _league(), _names(), _adp())
        client = app.test_client()

        client.post("/draft/1")
        client.post("/draft/2")

        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert "Mike Trout" not in html
        assert "Gerrit Cole" not in html

        resp = client.post("/reset")
        assert resp.status_code == 200

        resp = client.get("/")
        html = resp.get_data(as_text=True)
        assert "Mike Trout" in html
        assert "Gerrit Cole" in html
        assert "Pete Alonso" in html
