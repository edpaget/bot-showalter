from typing import TYPE_CHECKING

from fantasy_baseball_manager.config_toml import WebConfig
from fantasy_baseball_manager.domain.yahoo_league import YahooDraftSetupInfo, YahooLeagueInfo
from fantasy_baseball_manager.web.types import WebConfigType, YahooDraftSetupInfoType, YahooLeagueInfoType

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


_SAMPLE_INFO = YahooLeagueInfo(
    league_key="449.l.12345",
    league_name="Test Fantasy League",
    season=2026,
    num_teams=12,
    is_keeper=True,
    max_keepers=5,
    user_team_name="My Team",
)


class TestYahooLeagueInfoType:
    def test_from_domain_round_trip(self) -> None:
        result = YahooLeagueInfoType.from_domain(_SAMPLE_INFO)
        assert result.league_key == "449.l.12345"
        assert result.league_name == "Test Fantasy League"
        assert result.season == 2026
        assert result.num_teams == 12
        assert result.is_keeper is True
        assert result.max_keepers == 5
        assert result.user_team_name == "My Team"

    def test_from_domain_none_fields(self) -> None:
        info = YahooLeagueInfo(
            league_key="449.l.99",
            league_name="Redraft League",
            season=2026,
            num_teams=10,
            is_keeper=False,
            max_keepers=None,
            user_team_name=None,
        )
        result = YahooLeagueInfoType.from_domain(info)
        assert result.is_keeper is False
        assert result.max_keepers is None
        assert result.user_team_name is None


class TestWebConfigTypeWithYahooLeague:
    def test_with_yahoo_league_info(self) -> None:
        config = WebConfig()
        result = WebConfigType.from_domain(config, yahoo_league_info=_SAMPLE_INFO)
        assert result.yahoo_league is not None
        assert result.yahoo_league.league_name == "Test Fantasy League"

    def test_without_yahoo_league_info(self) -> None:
        config = WebConfig()
        result = WebConfigType.from_domain(config)
        assert result.yahoo_league is None


class TestYahooTeamsQuery:
    _QUERY = """
        query YahooTeams($leagueKey: String!) {
            yahooTeams(leagueKey: $leagueKey) {
                teamKey
                name
                managerName
                isOwnedByUser
            }
        }
    """

    def test_returns_teams(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345"}},
        )
        assert response.status_code == 200
        data = response.json()["data"]["yahooTeams"]
        assert len(data) == 2
        dynasty = next(t for t in data if t["teamKey"] == "449.l.12345.t.1")
        assert dynasty["name"] == "Dynasty Kings"
        assert dynasty["managerName"] == "Alice"
        assert dynasty["isOwnedByUser"] is True
        rival = next(t for t in data if t["teamKey"] == "449.l.12345.t.2")
        assert rival["isOwnedByUser"] is False

    def test_errors_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345"}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "Yahoo league is not configured" in body["errors"][0]["message"]


class TestYahooStandingsQuery:
    _QUERY = """
        query YahooStandings($leagueKey: String!, $season: Int!) {
            yahooStandings(leagueKey: $leagueKey, season: $season) {
                teamKey
                teamName
                finalRank
                statValues
            }
        }
    """

    def test_returns_standings_sorted_by_rank(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"leagueKey": "449.l.12345", "season": 2026},
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["yahooStandings"]
        assert len(data) == 2
        assert data[0]["finalRank"] == 1
        assert data[0]["teamName"] == "Dynasty Kings"
        assert data[0]["statValues"]["HR"] == 250.0
        assert data[1]["finalRank"] == 2
        assert data[1]["teamName"] == "Rival Squad"

    def test_returns_empty_for_unknown_season(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"leagueKey": "449.l.12345", "season": 2020},
            },
        )
        assert response.status_code == 200
        assert response.json()["data"]["yahooStandings"] == []

    def test_errors_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"leagueKey": "449.l.12345", "season": 2026},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "Yahoo league is not configured" in body["errors"][0]["message"]


class TestYahooRostersQuery:
    _QUERY = """
        query YahooRosters($leagueKey: String!) {
            yahooRosters(leagueKey: $leagueKey) {
                teamKey
                leagueKey
                season
                week
                asOf
                entries {
                    yahooPlayerKey
                    playerName
                    position
                    acquisitionType
                    playerId
                }
            }
        }
    """

    def test_returns_rosters_for_all_teams(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345"}},
        )
        assert response.status_code == 200
        data = response.json()["data"]["yahooRosters"]
        assert len(data) == 2
        team_keys = {r["teamKey"] for r in data}
        assert team_keys == {"449.l.12345.t.1", "449.l.12345.t.2"}

    def test_roster_entries_have_correct_fields(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345"}},
        )
        data = response.json()["data"]["yahooRosters"]
        team1 = next(r for r in data if r["teamKey"] == "449.l.12345.t.1")
        assert len(team1["entries"]) == 2
        assert team1["season"] == 2026
        assert team1["week"] == 1
        assert team1["asOf"] == "2026-03-28"

        mapped = next(e for e in team1["entries"] if e["playerName"] == "Mike Trout")
        assert mapped["playerId"] == 1
        assert mapped["position"] == "OF"
        assert mapped["acquisitionType"] == "draft"

        unmapped = next(e for e in team1["entries"] if e["playerName"] == "Unknown Prospect")
        assert unmapped["playerId"] is None

    def test_returns_empty_for_unknown_league(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "999.l.99999"}},
        )
        assert response.status_code == 200
        assert response.json()["data"]["yahooRosters"] == []

    def test_errors_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345"}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "Yahoo league is not configured" in body["errors"][0]["message"]


class TestYahooRosterQuery:
    _QUERY = """
        query YahooRoster($teamKey: String!, $leagueKey: String!) {
            yahooRoster(teamKey: $teamKey, leagueKey: $leagueKey) {
                teamKey
                season
                week
                asOf
                entries {
                    yahooPlayerKey
                    playerName
                    position
                    acquisitionType
                    playerId
                }
            }
        }
    """

    def test_returns_single_team_roster(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"teamKey": "449.l.12345.t.2", "leagueKey": "449.l.12345"},
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["yahooRoster"]
        assert data["teamKey"] == "449.l.12345.t.2"
        assert len(data["entries"]) == 1
        assert data["entries"][0]["playerName"] == "Shohei Ohtani"
        assert data["entries"][0]["playerId"] == 2

    def test_returns_null_for_unknown_team(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"teamKey": "449.l.12345.t.99", "leagueKey": "449.l.12345"},
            },
        )
        assert response.status_code == 200
        assert response.json()["data"]["yahooRoster"] is None

    def test_errors_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"teamKey": "449.l.12345.t.1", "leagueKey": "449.l.12345"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "Yahoo league is not configured" in body["errors"][0]["message"]


class TestDeriveKeeperCostsMutation:
    _MUTATION = """
        mutation DeriveKeeperCosts($leagueKey: String!, $season: Int!) {
            deriveKeeperCosts(leagueKey: $leagueKey, season: $season)
        }
    """

    def test_derives_best_n_costs(self, yahoo_keeper_client: TestClient) -> None:
        response = yahoo_keeper_client.post(
            "/graphql",
            json={
                "query": self._MUTATION,
                "variables": {"leagueKey": "449.l.12345", "season": 2026},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" not in body, body.get("errors")
        count = body["data"]["deriveKeeperCosts"]
        assert count >= 1

    def test_errors_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": self._MUTATION,
                "variables": {"leagueKey": "449.l.12345", "season": 2026},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "Yahoo league is not configured" in body["errors"][0]["message"]


class TestYahooKeeperOverviewQuery:
    _QUERY = """
        query YahooKeeperOverview($leagueKey: String!, $season: Int!, $maxKeepers: Int!) {
            yahooKeeperOverview(leagueKey: $leagueKey, season: $season, maxKeepers: $maxKeepers) {
                teamProjections {
                    teamKey
                    teamName
                    isUser
                    totalValue
                    categoryTotals
                    keepers {
                        playerId
                        playerName
                        position
                        value
                        categoryScores
                    }
                }
                tradeTargets {
                    playerId
                    playerName
                    position
                    value
                    owningTeamName
                    owningTeamKey
                    rankOnTeam
                }
                categoryNames
            }
        }
    """

    def test_returns_team_projections(self, yahoo_keeper_client: TestClient) -> None:
        response = yahoo_keeper_client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"leagueKey": "449.l.12345", "season": 2026, "maxKeepers": 2},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" not in body, body.get("errors")
        overview = body["data"]["yahooKeeperOverview"]
        assert len(overview["teamProjections"]) == 2
        # Rosters come from prior season, so team keys are 448.*
        for proj in overview["teamProjections"]:
            assert proj["teamKey"] in ("448.l.12345.t.1", "448.l.12345.t.2")
            assert len(proj["keepers"]) >= 1

    def test_identifies_user_team(self, yahoo_keeper_client: TestClient) -> None:
        response = yahoo_keeper_client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"leagueKey": "449.l.12345", "season": 2026, "maxKeepers": 2},
            },
        )
        overview = response.json()["data"]["yahooKeeperOverview"]
        user_proj = next(p for p in overview["teamProjections"] if p["isUser"])
        assert user_proj["teamKey"] == "448.l.12345.t.1"

    def test_errors_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"leagueKey": "449.l.12345", "season": 2026, "maxKeepers": 2},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "Yahoo league is not configured" in body["errors"][0]["message"]


class TestPlanKeeperDraftAutoDerive:
    """planKeeperDraft should auto-derive costs from Yahoo when no planner is pre-built."""

    _QUERY = """
        query PlanKeeperDraft($season: Int!, $maxKeepers: Int!) {
            planKeeperDraft(season: $season, maxKeepers: $maxKeepers) {
                scenarios {
                    keeperIds
                    totalSurplus
                }
            }
        }
    """

    def test_auto_derives_from_yahoo(self, yahoo_keeper_client: TestClient) -> None:
        """No explicit deriveKeeperCosts call — planKeeperDraft should derive automatically."""
        response = yahoo_keeper_client.post(
            "/graphql",
            json={
                "query": self._QUERY,
                "variables": {"season": 2026, "maxKeepers": 2},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" not in body, body.get("errors")
        data = body["data"]["planKeeperDraft"]
        assert len(data["scenarios"]) >= 1


class TestWebConfigQueryYahooLeague:
    def test_returns_null_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        webConfig {
                            projections { system version }
                            valuations { system version }
                            yahooLeague {
                                leagueKey
                                leagueName
                                season
                                numTeams
                                isKeeper
                                maxKeepers
                                userTeamName
                            }
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["webConfig"]
        assert data["yahooLeague"] is None


class TestYahooDraftSetupInfoType:
    def test_from_domain_round_trip(self) -> None:
        info = YahooDraftSetupInfo(
            num_teams=10,
            draft_format="auction",
            user_team_id=3,
            team_names={1: "Team A", 2: "Team B", 3: "Team C"},
            draft_order=[],
            is_keeper=False,
            max_keepers=None,
            keeper_player_ids=[],
        )
        result = YahooDraftSetupInfoType.from_domain(info)
        assert result.num_teams == 10
        assert result.draft_format == "auction"
        assert result.user_team_id == 3
        assert result.is_keeper is False
        assert result.max_keepers is None
        assert result.keeper_player_ids == []
        assert result.draft_order == []

    def test_with_keeper_data(self) -> None:
        info = YahooDraftSetupInfo(
            num_teams=12,
            draft_format="snake",
            user_team_id=1,
            team_names={1: "My Team", 2: "Other"},
            draft_order=[2, 1],
            is_keeper=True,
            max_keepers=5,
            keeper_player_ids=[100, 200, 300],
        )
        result = YahooDraftSetupInfoType.from_domain(info)
        assert result.is_keeper is True
        assert result.max_keepers == 5
        assert result.keeper_player_ids == [100, 200, 300]
        assert result.draft_order == [2, 1]


class TestYahooDraftSetupQuery:
    _QUERY = """
        query YahooDraftSetup($leagueKey: String!, $season: Int!) {
            yahooDraftSetup(leagueKey: $leagueKey, season: $season) {
                numTeams
                draftFormat
                userTeamId
                teamNames
                draftOrder
                isKeeper
                maxKeepers
                keeperPlayerIds
            }
        }
    """

    def test_returns_setup_for_synced_league(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345", "season": 2026}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" not in body, body.get("errors")
        data = body["data"]["yahooDraftSetup"]
        assert data["numTeams"] == 12
        assert data["draftFormat"] == "live"
        assert data["userTeamId"] == 1
        assert data["teamNames"] == {"1": "Dynasty Kings", "2": "Rival Squad"}
        assert data["draftOrder"] == []
        assert data["isKeeper"] is True
        assert data["keeperPlayerIds"] == []

    def test_returns_keeper_ids_when_costs_exist(self, yahoo_keeper_client: TestClient) -> None:
        response = yahoo_keeper_client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345", "season": 2026}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" not in body, body.get("errors")
        data = body["data"]["yahooDraftSetup"]
        assert data["isKeeper"] is True
        assert data["maxKeepers"] == 5

    def test_returns_empty_keepers_for_non_keeper_league(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345", "season": 2026}},
        )
        assert response.status_code == 200
        data = response.json()["data"]["yahooDraftSetup"]
        assert data["keeperPlayerIds"] == []

    def test_errors_when_not_configured(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "449.l.12345", "season": 2026}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "Yahoo league is not configured" in body["errors"][0]["message"]

    def test_errors_when_league_not_synced(self, yahoo_client: TestClient) -> None:
        response = yahoo_client.post(
            "/graphql",
            json={"query": self._QUERY, "variables": {"leagueKey": "999.l.99999", "season": 2026}},
        )
        assert response.status_code == 200
        body = response.json()
        assert "errors" in body
        assert "not been synced" in body["errors"][0]["message"]
