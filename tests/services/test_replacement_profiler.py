import pytest

from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    StatType,
)
from fantasy_baseball_manager.domain.replacement_profile import ReplacementProfile
from fantasy_baseball_manager.models.zar.engine import (
    compute_replacement_level,
    compute_z_scores,
    convert_rate_stats,
)
from fantasy_baseball_manager.services.replacement_profiler import (
    compute_replacement_profiles,
)


def _counting(key: str, direction: Direction = Direction.HIGHER) -> CategoryConfig:
    return CategoryConfig(key=key, name=key, stat_type=StatType.COUNTING, direction=direction)


def _rate(
    key: str,
    numerator: str,
    denominator: str,
    direction: Direction = Direction.HIGHER,
) -> CategoryConfig:
    return CategoryConfig(
        key=key,
        name=key,
        stat_type=StatType.RATE,
        direction=direction,
        numerator=numerator,
        denominator=denominator,
    )


class TestComputeReplacementProfiles:
    def test_correct_replacement_player_identified(self) -> None:
        """5 OF-eligible players, 2 spots × 2 teams = 4 draftable, 5th best is replacement."""
        stats = [
            {"hr": 40.0, "rbi": 120.0},
            {"hr": 35.0, "rbi": 100.0},
            {"hr": 30.0, "rbi": 90.0},
            {"hr": 25.0, "rbi": 80.0},
            {"hr": 15.0, "rbi": 50.0},  # replacement
        ]
        positions = [["OF"]] * 5
        categories = [_counting("hr"), _counting("rbi")]
        roster_spots = {"OF": 2}

        result = compute_replacement_profiles(
            stats_list=stats,
            position_map=positions,
            roster_spots=roster_spots,
            num_teams=2,
            categories=categories,
            player_type=PlayerType.BATTER,
        )

        assert "OF" in result
        profile = result["OF"]
        assert isinstance(profile, ReplacementProfile)
        assert profile.position == "OF"
        assert profile.player_type == "batter"
        assert profile.stat_line == stats[4]

    def test_multiple_positions(self) -> None:
        """Separate pools at 1B and OF get independent replacement players."""
        stats = [
            {"hr": 40.0},  # 1B
            {"hr": 30.0},  # 1B
            {"hr": 20.0},  # 1B — replacement for 1B
            {"hr": 35.0},  # OF
            {"hr": 25.0},  # OF
            {"hr": 10.0},  # OF — replacement for OF
        ]
        positions = [["1B"], ["1B"], ["1B"], ["OF"], ["OF"], ["OF"]]
        categories = [_counting("hr")]
        roster_spots = {"1B": 1, "OF": 1}

        result = compute_replacement_profiles(
            stats_list=stats,
            position_map=positions,
            roster_spots=roster_spots,
            num_teams=2,
            categories=categories,
            player_type=PlayerType.BATTER,
        )

        assert result["1B"].stat_line == stats[2]
        assert result["OF"].stat_line == stats[5]

    def test_multi_position_eligibility(self) -> None:
        """A player eligible at both positions appears in both pools."""
        stats = [
            {"hr": 40.0},  # 1B/OF
            {"hr": 30.0},  # 1B
            {"hr": 20.0},  # OF
        ]
        positions = [["1B", "OF"], ["1B"], ["OF"]]
        categories = [_counting("hr")]
        roster_spots = {"1B": 1, "OF": 1}

        result = compute_replacement_profiles(
            stats_list=stats,
            position_map=positions,
            roster_spots=roster_spots,
            num_teams=1,
            categories=categories,
            player_type=PlayerType.BATTER,
        )

        # 1B pool: players 0 (40) and 1 (30), 1 spot × 1 team = 1 draftable, replacement = player 1
        assert result["1B"].stat_line == stats[1]
        # OF pool: players 0 (40) and 2 (20), 1 spot × 1 team = 1 draftable, replacement = player 2
        assert result["OF"].stat_line == stats[2]

    def test_thin_position(self) -> None:
        """Only 2 players at C but draftable=6, worst of 2 is replacement."""
        stats = [
            {"hr": 20.0},
            {"hr": 10.0},
        ]
        positions = [["C"], ["C"]]
        categories = [_counting("hr")]
        roster_spots = {"C": 3}

        result = compute_replacement_profiles(
            stats_list=stats,
            position_map=positions,
            roster_spots=roster_spots,
            num_teams=2,
            categories=categories,
            player_type=PlayerType.BATTER,
        )

        # draftable=6 > 2 players, so worst eligible (player 1) is replacement
        assert result["C"].stat_line == stats[1]

    def test_no_eligible_players(self) -> None:
        """Position in roster_spots but no players eligible → zero-stat profile."""
        stats = [{"hr": 30.0}]
        positions = [["OF"]]
        categories = [_counting("hr"), _counting("rbi")]
        roster_spots = {"OF": 1, "C": 1}

        result = compute_replacement_profiles(
            stats_list=stats,
            position_map=positions,
            roster_spots=roster_spots,
            num_teams=1,
            categories=categories,
            player_type=PlayerType.BATTER,
        )

        assert result["C"].stat_line == {"hr": 0.0, "rbi": 0.0}

    def test_empty_projections(self) -> None:
        """Empty projections returns empty dict."""
        categories = [_counting("hr")]
        roster_spots = {"OF": 1}

        result = compute_replacement_profiles(
            stats_list=[],
            position_map=[],
            roster_spots=roster_spots,
            num_teams=1,
            categories=categories,
            player_type=PlayerType.BATTER,
        )

        assert result == {}

    def test_original_stats_returned_not_converted(self) -> None:
        """Rate stat in profile is the raw value, not the marginal contribution."""
        stats = [
            {"h": 180.0, "ab": 600.0},  # .300
            {"h": 150.0, "ab": 600.0},  # .250
            {"h": 120.0, "ab": 600.0},  # .200 — replacement
        ]
        positions = [["OF"]] * 3
        categories = [_rate("avg", "h", "ab")]
        roster_spots = {"OF": 1}

        result = compute_replacement_profiles(
            stats_list=stats,
            position_map=positions,
            roster_spots=roster_spots,
            num_teams=2,
            categories=categories,
            player_type=PlayerType.BATTER,
        )

        # Should return original stats, not converted marginal values
        profile = result["OF"]
        assert profile.stat_line["h"] == 120.0
        assert profile.stat_line["ab"] == 600.0

    def test_consistent_with_engine(self) -> None:
        """Replacement player's composite z matches the engine's threshold."""
        stats = [
            {"hr": 40.0, "rbi": 120.0},
            {"hr": 35.0, "rbi": 100.0},
            {"hr": 30.0, "rbi": 90.0},
            {"hr": 25.0, "rbi": 80.0},
            {"hr": 15.0, "rbi": 50.0},
        ]
        positions = [["OF"]] * 5
        categories = [_counting("hr"), _counting("rbi")]
        category_keys = [c.key for c in categories]
        roster_spots = {"OF": 2}
        num_teams = 2

        # Get replacement profiles
        profiles = compute_replacement_profiles(
            stats_list=stats,
            position_map=positions,
            roster_spots=roster_spots,
            num_teams=num_teams,
            categories=categories,
            player_type=PlayerType.BATTER,
        )

        # Run engine to get replacement level
        converted = convert_rate_stats(stats, categories)
        z_scores = compute_z_scores(converted, category_keys)
        engine_replacement = compute_replacement_level(z_scores, positions, roster_spots, num_teams)

        # Find replacement player index from profile stat line
        repl_stat_line = profiles["OF"].stat_line
        repl_index = stats.index(repl_stat_line)

        # The replacement player's composite z should match the engine's threshold
        assert z_scores[repl_index].composite_z == pytest.approx(engine_replacement["OF"])
