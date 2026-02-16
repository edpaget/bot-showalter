from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.services.cohort import (
    assign_age_cohorts,
    assign_experience_cohorts,
    assign_top300_cohorts,
)


class TestAssignAgeCohorts:
    def test_age_young(self) -> None:
        players = {1: Player(name_first="A", name_last="B", id=1, birth_date="2000-03-15")}
        result = assign_age_cohorts(players, season=2025)
        assert result[1] == "young"

    def test_age_prime(self) -> None:
        players = {1: Player(name_first="A", name_last="B", id=1, birth_date="1994-01-01")}
        result = assign_age_cohorts(players, season=2025)
        assert result[1] == "prime"

    def test_age_veteran(self) -> None:
        players = {1: Player(name_first="A", name_last="B", id=1, birth_date="1993-01-01")}
        result = assign_age_cohorts(players, season=2025)
        assert result[1] == "veteran"

    def test_age_unknown(self) -> None:
        players = {1: Player(name_first="A", name_last="B", id=1, birth_date=None)}
        result = assign_age_cohorts(players, season=2025)
        assert result[1] == "unknown"

    def test_age_boundary_july1(self) -> None:
        # Born 1999-07-01, season 2025 → age on July 1 = 2025-1999-0 = 26 → "prime"
        players = {1: Player(name_first="A", name_last="B", id=1, birth_date="1999-07-01")}
        result = assign_age_cohorts(players, season=2025)
        assert result[1] == "prime"

    def test_age_boundary_july2(self) -> None:
        # Born 1999-07-02, season 2025 → age on July 1 = 2025-1999-1 = 25 → "young"
        players = {1: Player(name_first="A", name_last="B", id=1, birth_date="1999-07-02")}
        result = assign_age_cohorts(players, season=2025)
        assert result[1] == "young"


class TestAssignExperienceCohorts:
    def test_experience_rookie(self) -> None:
        prior_batting = [
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=150),
        ]
        result = assign_experience_cohorts(prior_batting, player_ids={1})
        assert result[1] == "rookie"

    def test_experience_limited(self) -> None:
        prior_batting = [
            BattingStats(player_id=1, season=2023, source="fangraphs", pa=400),
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=400),
        ]
        result = assign_experience_cohorts(prior_batting, player_ids={1})
        assert result[1] == "limited"

    def test_experience_established(self) -> None:
        prior_batting = [
            BattingStats(player_id=1, season=2022, source="fangraphs", pa=1000),
            BattingStats(player_id=1, season=2023, source="fangraphs", pa=1000),
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=1000),
        ]
        result = assign_experience_cohorts(prior_batting, player_ids={1})
        assert result[1] == "established"

    def test_experience_no_prior_stats(self) -> None:
        result = assign_experience_cohorts([], player_ids={1})
        assert result[1] == "rookie"


class TestAssignTop300Cohorts:
    def test_top300_splits(self) -> None:
        actuals = [BattingStats(player_id=i, season=2025, source="fangraphs", war=float(5 - i)) for i in range(1, 6)]
        result = assign_top300_cohorts(actuals, top_n=3)
        assert result[1] == "top300"
        assert result[2] == "top300"
        assert result[3] == "top300"
        assert result[4] == "rest"
        assert result[5] == "rest"

    def test_top300_null_war(self) -> None:
        actuals = [
            BattingStats(player_id=1, season=2025, source="fangraphs", war=2.0),
            BattingStats(player_id=2, season=2025, source="fangraphs", war=None),
        ]
        result = assign_top300_cohorts(actuals, top_n=1)
        assert result[1] == "top300"
        assert result[2] == "rest"
