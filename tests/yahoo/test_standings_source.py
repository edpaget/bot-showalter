from typing import Any

from fantasy_baseball_manager.yahoo.standings_source import STAT_ID_MAP, YahooStandingsSource


def _build_team(
    team_key: str,
    name: str,
    rank: int,
    stats: list[dict[str, str]],
) -> dict[str, Any]:
    """Build a single team entry matching the Yahoo API structure."""
    team_info: list[Any] = [
        {"team_key": team_key},
        {"team_id": "1"},
        {"name": name},
        [],
    ]
    stat_entries = [{"stat": s} for s in stats]
    return {
        "team": [
            team_info,
            {"team_stats": {"coverage_type": "season", "season": "2025", "stats": stat_entries}},
            {"team_standings": {"rank": rank}},
        ]
    }


def _build_response(*teams: dict[str, Any]) -> dict[str, Any]:
    """Wrap team dicts into a full Yahoo API response."""
    teams_section: dict[str, Any] = {"count": len(teams)}
    for i, team in enumerate(teams):
        teams_section[str(i)] = team
    return {
        "fantasy_content": {
            "league": [
                {"league_key": "458.l.135575", "season": "2025"},
                {"standings": [{"teams": teams_section}]},
            ]
        }
    }


class TestYahooStandingsSource:
    def test_happy_path(self) -> None:
        team = _build_team(
            "458.l.135575.t.10",
            "My Team",
            rank=1,
            stats=[
                {"stat_id": "7", "value": "764"},
                {"stat_id": "12", "value": "198"},
                {"stat_id": "13", "value": "667"},
                {"stat_id": "16", "value": "184"},
                {"stat_id": "4", "value": ".340"},
                {"stat_id": "26", "value": "3.55"},
                {"stat_id": "27", "value": "1.15"},
                {"stat_id": "28", "value": "74"},
                {"stat_id": "42", "value": "1110"},
                {"stat_id": "50", "value": "1076.2"},
                {"stat_id": "90", "value": "65"},
            ],
        )
        response = _build_response(team)
        results = YahooStandingsSource._parse_standings(response, "458.l.135575", 2025)

        assert len(results) == 1
        r = results[0]
        assert r.team_key == "458.l.135575.t.10"
        assert r.league_key == "458.l.135575"
        assert r.season == 2025
        assert r.team_name == "My Team"
        assert r.final_rank == 1
        assert r.stat_values["r"] == 764.0
        assert r.stat_values["hr"] == 198.0
        assert r.stat_values["rbi"] == 667.0
        assert r.stat_values["sb"] == 184.0
        assert r.stat_values["obp"] == 0.340
        assert r.stat_values["era"] == 3.55
        assert r.stat_values["whip"] == 1.15
        assert r.stat_values["w"] == 74.0
        assert r.stat_values["so"] == 1110.0
        assert r.stat_values["ip"] == 1076.2
        assert r.stat_values["sv+hld"] == 65.0

    def test_stat_id_mapping_covers_all_fbm_categories(self) -> None:
        expected_keys = {"r", "hr", "rbi", "sb", "obp", "era", "whip", "so", "w", "sv+hld", "ip"}
        assert set(STAT_ID_MAP.values()) == expected_keys

    def test_unmapped_stat_ids_ignored(self) -> None:
        team = _build_team(
            "t.1",
            "Team A",
            rank=1,
            stats=[
                {"stat_id": "60", "value": ""},  # H/AB display stat
                {"stat_id": "999", "value": "42"},  # unknown stat
                {"stat_id": "7", "value": "100"},  # known: runs
            ],
        )
        response = _build_response(team)
        results = YahooStandingsSource._parse_standings(response, "lk", 2025)
        assert results[0].stat_values == {"r": 100.0}

    def test_empty_value_skipped(self) -> None:
        team = _build_team(
            "t.1",
            "Team A",
            rank=1,
            stats=[
                {"stat_id": "7", "value": ""},
                {"stat_id": "12", "value": "50"},
            ],
        )
        response = _build_response(team)
        results = YahooStandingsSource._parse_standings(response, "lk", 2025)
        assert "r" not in results[0].stat_values
        assert results[0].stat_values["hr"] == 50.0

    def test_empty_teams(self) -> None:
        response = _build_response()
        results = YahooStandingsSource._parse_standings(response, "lk", 2025)
        assert results == []

    def test_multiple_teams_parsed(self) -> None:
        team1 = _build_team("t.1", "First", rank=1, stats=[{"stat_id": "7", "value": "800"}])
        team2 = _build_team("t.2", "Second", rank=2, stats=[{"stat_id": "7", "value": "750"}])
        response = _build_response(team1, team2)
        results = YahooStandingsSource._parse_standings(response, "lk", 2025)
        assert len(results) == 2
        assert results[0].team_name == "First"
        assert results[1].team_name == "Second"
