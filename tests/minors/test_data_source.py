"""Tests for minors/data_source.py"""

from __future__ import annotations

from typing import Any

from fantasy_baseball_manager.minors.data_source import MLBStatsAPIDataSource
from fantasy_baseball_manager.minors.types import MinorLeagueLevel


class TestMLBStatsAPIDataSource:
    """Tests for MLBStatsAPIDataSource response parsing."""

    def test_parse_batting_response_empty(self) -> None:
        source = MLBStatsAPIDataSource()
        result = source._parse_batting_response({}, 2024, MinorLeagueLevel.AAA)
        assert result == []

    def test_parse_batting_response_no_splits(self) -> None:
        source = MLBStatsAPIDataSource()
        data: dict[str, Any] = {"stats": [{"splits": []}]}
        result = source._parse_batting_response(data, 2024, MinorLeagueLevel.AAA)
        assert result == []

    def test_parse_batting_response_single_player(self) -> None:
        source = MLBStatsAPIDataSource()
        data: dict[str, Any] = {
            "stats": [
                {
                    "splits": [
                        {
                            "player": {"id": 12345, "fullName": "Test Player", "currentAge": 23},
                            "team": {"name": "Syracuse Mets"},
                            "league": {"name": "International League"},
                            "stat": {
                                "plateAppearances": 500,
                                "atBats": 450,
                                "hits": 130,
                                "doubles": 25,
                                "triples": 5,
                                "homeRuns": 10,
                                "rbi": 60,
                                "runs": 70,
                                "baseOnBalls": 40,
                                "strikeOuts": 100,
                                "hitByPitch": 5,
                                "sacFlies": 5,
                                "stolenBases": 15,
                                "caughtStealing": 5,
                                "avg": ".289",
                                "obp": ".360",
                                "slg": ".440",
                            },
                        }
                    ]
                }
            ]
        }

        result = source._parse_batting_response(data, 2024, MinorLeagueLevel.AAA)

        assert len(result) == 1
        player = result[0]
        assert player.player_id == "12345"
        assert player.name == "Test Player"
        assert player.season == 2024
        assert player.age == 23
        assert player.level == MinorLeagueLevel.AAA
        assert player.team == "Syracuse Mets"
        assert player.league == "International League"
        assert player.pa == 500
        assert player.ab == 450
        assert player.h == 130
        assert player.singles == 90  # h - doubles - triples - hr
        assert player.doubles == 25
        assert player.triples == 5
        assert player.hr == 10
        assert player.rbi == 60
        assert player.r == 70
        assert player.bb == 40
        assert player.so == 100
        assert player.hbp == 5
        assert player.sf == 5
        assert player.sb == 15
        assert player.cs == 5
        assert player.avg == 0.289
        assert player.obp == 0.360
        assert player.slg == 0.440

    def test_parse_batting_response_skips_missing_player_id(self) -> None:
        source = MLBStatsAPIDataSource()
        data: dict[str, Any] = {
            "stats": [
                {
                    "splits": [
                        {
                            "player": {"fullName": "No ID Player"},
                            "team": {"name": "Team"},
                            "league": {"name": "League"},
                            "stat": {"plateAppearances": 100},
                        }
                    ]
                }
            ]
        }

        result = source._parse_batting_response(data, 2024, MinorLeagueLevel.AAA)
        assert result == []

    def test_parse_pitching_response_single_player(self) -> None:
        source = MLBStatsAPIDataSource()
        data: dict[str, Any] = {
            "stats": [
                {
                    "splits": [
                        {
                            "player": {"id": 54321, "fullName": "Test Pitcher", "currentAge": 24},
                            "team": {"name": "Binghamton Rumble Ponies"},
                            "league": {"name": "Eastern League"},
                            "stat": {
                                "gamesPlayed": 25,
                                "gamesStarted": 25,
                                "inningsPitched": "140.1",
                                "wins": 10,
                                "losses": 5,
                                "saves": 0,
                                "hits": 120,
                                "runs": 55,
                                "earnedRuns": 50,
                                "homeRuns": 12,
                                "baseOnBalls": 40,
                                "strikeOuts": 150,
                                "hitByPitch": 5,
                                "era": "3.21",
                                "whip": "1.14",
                            },
                        }
                    ]
                }
            ]
        }

        result = source._parse_pitching_response(data, 2024, MinorLeagueLevel.AA)

        assert len(result) == 1
        pitcher = result[0]
        assert pitcher.player_id == "54321"
        assert pitcher.name == "Test Pitcher"
        assert pitcher.season == 2024
        assert pitcher.age == 24
        assert pitcher.level == MinorLeagueLevel.AA
        assert pitcher.team == "Binghamton Rumble Ponies"
        assert pitcher.league == "Eastern League"
        assert pitcher.g == 25
        assert pitcher.gs == 25
        assert abs(pitcher.ip - 140.333) < 0.01  # 140.1 = 140 + 1/3
        assert pitcher.w == 10
        assert pitcher.losses == 5
        assert pitcher.sv == 0
        assert pitcher.h == 120
        assert pitcher.r == 55
        assert pitcher.er == 50
        assert pitcher.hr == 12
        assert pitcher.bb == 40
        assert pitcher.so == 150
        assert pitcher.hbp == 5
        assert pitcher.era == 3.21
        assert pitcher.whip == 1.14

    def test_parse_ip_formats(self) -> None:
        source = MLBStatsAPIDataSource()
        # Standard format: x.y where y is thirds
        assert source._parse_ip("6.0") == 6.0
        assert abs(source._parse_ip("6.1") - 6.333) < 0.01
        assert abs(source._parse_ip("6.2") - 6.667) < 0.01
        # Integer
        assert source._parse_ip(6) == 6.0
        # Float passthrough
        assert source._parse_ip(6.5) == 6.5
        # None
        assert source._parse_ip(None) == 0.0

    def test_parse_float_formats(self) -> None:
        source = MLBStatsAPIDataSource()
        assert source._parse_float(".300") == 0.300
        assert source._parse_float("3.21") == 3.21
        assert source._parse_float(0.300) == 0.300
        assert source._parse_float(None) == 0.0
        assert source._parse_float("invalid") == 0.0

    def test_parse_age_formats(self) -> None:
        source = MLBStatsAPIDataSource()
        assert source._parse_age(23) == 23
        assert source._parse_age("23") == 23
        assert source._parse_age(None) == 0
        assert source._parse_age("invalid") == 0
