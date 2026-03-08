from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestBoardQuery:
    def test_returns_all_rows(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        board(season: 2026) {
                            rows {
                                playerId
                                playerName
                                rank
                                playerType
                                position
                                value
                                categoryZScores
                            }
                            battingCategories
                            pitchingCategories
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["board"]
        assert len(data["rows"]) == 3
        assert data["battingCategories"] == ["HR", "RBI"]
        assert data["pitchingCategories"] == ["W", "K"]

        # Rows are sorted by value descending
        values = [r["value"] for r in data["rows"]]
        assert values == sorted(values, reverse=True)

        # First row is Mike Trout
        first = data["rows"][0]
        assert first["playerName"] == "Mike Trout"
        assert first["playerType"] == "batter"
        assert first["position"] == "OF"

    def test_filter_by_player_type(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        board(season: 2026, playerType: "pitcher") {
                            rows { playerName playerType }
                        }
                    }
                """
            },
        )
        data = response.json()["data"]["board"]
        assert len(data["rows"]) == 1
        assert data["rows"][0]["playerName"] == "Gerrit Cole"
        assert data["rows"][0]["playerType"] == "pitcher"

    def test_filter_by_position(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        board(season: 2026, position: "OF") {
                            rows { playerName position }
                        }
                    }
                """
            },
        )
        data = response.json()["data"]["board"]
        assert len(data["rows"]) == 2
        assert all(r["position"] == "OF" for r in data["rows"])

    def test_top_n(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        board(season: 2026, top: 2) {
                            rows { playerName }
                        }
                    }
                """
            },
        )
        data = response.json()["data"]["board"]
        assert len(data["rows"]) == 2

    def test_category_z_scores_included(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        board(season: 2026, playerType: "batter", top: 1) {
                            rows { categoryZScores }
                        }
                    }
                """
            },
        )
        data = response.json()["data"]["board"]
        scores = data["rows"][0]["categoryZScores"]
        assert "HR" in scores
        assert "RBI" in scores


class TestTiersQuery:
    def test_returns_tier_assignments(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        tiers(season: 2026) {
                            playerId
                            playerName
                            position
                            tier
                            value
                            rank
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["tiers"]
        assert len(data) > 0
        # Each tier entry has required fields
        for entry in data:
            assert "playerId" in entry
            assert "playerName" in entry
            assert "position" in entry
            assert "tier" in entry
            assert entry["tier"] >= 1

    def test_filter_by_player_type(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        tiers(season: 2026, playerType: "pitcher") {
                            playerName
                            position
                        }
                    }
                """
            },
        )
        data = response.json()["data"]["tiers"]
        assert len(data) == 1
        assert data[0]["position"] == "SP"


class TestScarcityQuery:
    def test_returns_per_position_metrics(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        scarcity(season: 2026) {
                            position
                            tier1Value
                            replacementValue
                            totalSurplus
                            dropoffSlope
                            steepRank
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["scarcity"]
        # We have OF and SP positions with valuations
        positions = {s["position"] for s in data}
        assert len(positions) > 0
        for entry in data:
            assert "tier1Value" in entry
            assert "replacementValue" in entry
            assert "dropoffSlope" in entry


class TestLeagueQuery:
    def test_returns_league_settings(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        league {
                            name
                            format
                            teams
                            budget
                            rosterBatters
                            rosterPitchers
                            rosterUtil
                            battingCategories { key name statType direction }
                            pitchingCategories { key name statType direction }
                            positions
                            pitcherPositions
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["league"]
        assert data["name"] == "Test League"
        assert data["format"] == "h2h_categories"
        assert data["teams"] == 10
        assert data["budget"] == 260
        assert data["rosterBatters"] == 14
        assert data["rosterPitchers"] == 9
        assert len(data["battingCategories"]) == 2
        assert len(data["pitchingCategories"]) == 2
        assert data["battingCategories"][0]["key"] == "HR"
        assert data["positions"]["C"] == 1
        assert data["pitcherPositions"]["SP"] == 5
