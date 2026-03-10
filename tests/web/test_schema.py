from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestProjectionsQuery:
    def test_lookup_by_player_name(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        projections(season: 2026, playerName: "Mike Trout") {
                            playerName
                            system
                            version
                            sourceType
                            playerType
                            stats
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["projections"]
        assert len(data) >= 1
        proj = data[0]
        assert proj["playerName"] == "Mike Trout"
        assert proj["system"] == "steamer"
        assert proj["playerType"] == "batter"
        assert "hr" in proj["stats"]

    def test_filter_by_system(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        projections(season: 2026, playerName: "Trout", system: "steamer") {
                            playerName
                            system
                        }
                    }
                """
            },
        )
        data = response.json()["data"]["projections"]
        assert all(p["system"] == "steamer" for p in data)


class TestValuationsQuery:
    def test_rankings_returns_sorted(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        valuations(season: 2026) {
                            playerName
                            system
                            version
                            projectionSystem
                            projectionVersion
                            playerType
                            position
                            value
                            rank
                            categoryScores
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["valuations"]
        assert len(data) == 3
        ranks = [v["rank"] for v in data]
        assert ranks == sorted(ranks)

    def test_filter_by_player_type(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        valuations(season: 2026, playerType: "pitcher") {
                            playerName
                            playerType
                        }
                    }
                """
            },
        )
        data = response.json()["data"]["valuations"]
        assert len(data) == 1
        assert data[0]["playerType"] == "pitcher"

    def test_filter_by_position(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        valuations(season: 2026, position: "OF") {
                            playerName
                            position
                        }
                    }
                """
            },
        )
        data = response.json()["data"]["valuations"]
        assert len(data) == 2
        assert all(v["position"] == "OF" for v in data)

    def test_top_n(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        valuations(season: 2026, top: 2) {
                            playerName
                        }
                    }
                """
            },
        )
        data = response.json()["data"]["valuations"]
        assert len(data) == 2


class TestADPReportQuery:
    def test_returns_report_sections(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        adpReport(season: 2026) {
                            season
                            system
                            version
                            provider
                            buyTargets {
                                playerId
                                playerName
                                playerType
                                position
                                zarRank
                                zarValue
                                adpRank
                                adpPick
                                rankDelta
                                provider
                            }
                            avoidList {
                                playerName
                                rankDelta
                            }
                            unrankedValuable {
                                playerName
                            }
                            nMatched
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["adpReport"]
        assert data["season"] == 2026
        assert data["system"] == "zar"
        assert data["provider"] == "fantasypros"
        assert isinstance(data["buyTargets"], list)
        assert isinstance(data["avoidList"], list)
        assert isinstance(data["unrankedValuable"], list)
        assert data["nMatched"] >= 0


class TestPlayerSearchQuery:
    def test_search_returns_matching_players(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        playerSearch(name: "Trout", season: 2026) {
                            playerId
                            name
                            team
                            age
                            primaryPosition
                            bats
                            throws
                            experience
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["playerSearch"]
        assert len(data) >= 1
        player = data[0]
        assert "Trout" in player["name"]
        assert player["playerId"] == 1
        assert player["bats"] == "R"


class TestPlayerBioQuery:
    def test_lookup_by_player_id(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        playerBio(playerId: 1, season: 2026) {
                            playerId
                            name
                            team
                            age
                            primaryPosition
                            bats
                            throws
                            experience
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["playerBio"]
        assert data["playerId"] == 1
        assert data["name"] == "Mike Trout"
        assert data["bats"] == "R"

    def test_returns_null_for_unknown_player(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        playerBio(playerId: 9999, season: 2026) {
                            playerId
                            name
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["playerBio"]
        assert data is None


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
                        board(season: 2026, position: OF) {
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

    def test_breakout_bust_ranks_included(self, client: TestClient) -> None:
        response = client.post(
            "/graphql",
            json={
                "query": """
                    query {
                        board(season: 2026) {
                            rows { breakoutRank bustRank }
                        }
                    }
                """
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]["board"]
        # Without breakout predictions, ranks should be null
        for row in data["rows"]:
            assert row["breakoutRank"] is None
            assert row["bustRank"] is None

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
