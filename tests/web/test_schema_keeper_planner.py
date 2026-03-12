"""Integration tests for the planKeeperDraft GraphQL query."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.config_toml import WebConfig
from fantasy_baseball_manager.domain import (
    CategoryConfig,
    Direction,
    KeeperCost,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.repos import SqliteKeeperCostRepo
from fantasy_baseball_manager.services import PlayerEligibilityService
from fantasy_baseball_manager.services.keeper_planner import KeeperPlannerService
from fantasy_baseball_manager.web import create_app
from fantasy_baseball_manager.web.app import KeeperPlannerRef

_LEAGUE = LeagueSettings(
    name="Test League",
    format=LeagueFormat.H2H_CATEGORIES,
    teams=10,
    budget=260,
    roster_batters=14,
    roster_pitchers=9,
    batting_categories=(
        CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        CategoryConfig(key="RBI", name="RBI", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
    ),
    pitching_categories=(
        CategoryConfig(key="W", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
    ),
    positions={"C": 1, "1B": 1, "OF": 3},
    pitcher_positions={"SP": 5, "RP": 2},
)

PLAN_QUERY = """
query PlanKeeperDraft(
  $season: Int!
  $maxKeepers: Int!
  $customScenarios: [[Int!]!]
  $boardPreviewSize: Int
) {
  planKeeperDraft(
    season: $season
    maxKeepers: $maxKeepers
    customScenarios: $customScenarios
    boardPreviewSize: $boardPreviewSize
  ) {
    scenarios {
      keeperIds
      keepers {
        playerId
        playerName
        position
        cost
        surplus
        projectedValue
        recommendation
      }
      totalSurplus
      boardPreview {
        playerId
        playerName
        playerType
        position
        originalValue
        adjustedValue
        valueChange
      }
      scarcity {
        position
        tier1Value
        replacementValue
        totalSurplus
        dropoffSlope
      }
      categoryNeeds {
        category
        currentRank
        targetRank
      }
      strongestCategories
      weakestCategories
    }
  }
}
"""


@pytest.fixture
def planner_client(session_provider) -> TestClient:
    """Client with KeeperPlannerService configured."""
    container = AnalysisContainer(session_provider)

    # Seed keeper costs
    keeper_repo = SqliteKeeperCostRepo(session_provider)
    keeper_repo.upsert_batch(
        [
            KeeperCost(player_id=1, season=2026, league="default", cost=10.0, source="auction"),
            KeeperCost(player_id=2, season=2026, league="default", cost=15.0, source="auction"),
        ]
    )
    with session_provider.connection() as conn:
        conn.commit()

    valuations = container.valuation_repo.get_by_season(2026, system="zar", version="1.0")
    players = container.player_repo.get_by_ids([v.player_id for v in valuations])
    projections = container.projection_repo.get_by_season(2026)

    eligibility = PlayerEligibilityService(
        container.position_appearance_repo,
        pitching_stats_repo=container.pitching_stats_repo,
    )
    batter_positions = eligibility.get_batter_positions(2026, _LEAGUE)
    pitcher_ids = [p.player_id for p in projections if p.player_type == "pitcher"]
    pitcher_positions = eligibility.get_pitcher_positions(2026, _LEAGUE, pitcher_ids)

    planner = KeeperPlannerService(
        keeper_costs=keeper_repo.find_by_season_league(2026, "default"),
        valuations=valuations,
        players=players,
        projections=projections,
        league=_LEAGUE,
        batter_positions=batter_positions,
        pitcher_positions=pitcher_positions,
    )

    app = create_app(
        container,
        _LEAGUE,
        default_system="zar",
        default_version="1.0",
        web_config=WebConfig(),
        keeper_planner_ref=KeeperPlannerRef(planner=planner),
    )
    return TestClient(app)


class TestPlanKeeperDraft:
    def test_returns_scenarios_with_keeper_data(self, planner_client: TestClient) -> None:
        response = planner_client.post(
            "/graphql",
            json={
                "query": PLAN_QUERY,
                "variables": {"season": 2026, "maxKeepers": 2, "boardPreviewSize": 5},
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["planKeeperDraft"]
        assert len(data["scenarios"]) >= 1

        scenario = data["scenarios"][0]
        assert len(scenario["keeperIds"]) > 0
        assert len(scenario["keepers"]) > 0
        assert isinstance(scenario["totalSurplus"], float)
        assert isinstance(scenario["boardPreview"], list)
        assert isinstance(scenario["strongestCategories"], list)
        assert isinstance(scenario["weakestCategories"], list)

    def test_custom_scenarios_work(self, planner_client: TestClient) -> None:
        response = planner_client.post(
            "/graphql",
            json={
                "query": PLAN_QUERY,
                "variables": {
                    "season": 2026,
                    "maxKeepers": 1,
                    "customScenarios": [[2]],
                    "boardPreviewSize": 3,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["planKeeperDraft"]
        # Should include the custom scenario with player 2
        custom_found = any(2 in s["keeperIds"] for s in data["scenarios"])
        assert custom_found

    def test_error_when_planner_not_configured(self, client: TestClient) -> None:
        """Client without keeper_planner should raise an error."""
        response = client.post(
            "/graphql",
            json={
                "query": PLAN_QUERY,
                "variables": {"season": 2026, "maxKeepers": 2},
            },
        )
        assert response.status_code == 200
        errors = response.json().get("errors")
        assert errors is not None
        assert "not configured" in errors[0]["message"]
