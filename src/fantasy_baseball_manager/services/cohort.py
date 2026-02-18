import logging
from collections import defaultdict

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.player import Player

logger = logging.getLogger(__name__)


def assign_age_cohorts(players: dict[int, Player], season: int) -> dict[int, str]:
    """Assign players to age cohorts based on age as of July 1 of the season."""
    result: dict[int, str] = {}
    for player_id, player in players.items():
        if player.birth_date is None:
            result[player_id] = "unknown"
            continue
        birth_year = int(player.birth_date[:4])
        birth_month = int(player.birth_date[5:7])
        birth_day = int(player.birth_date[8:10])
        # Age as of July 1: subtract 1 if born after July 1
        age = season - birth_year - (1 if (birth_month, birth_day) > (7, 1) else 0)
        if age < 26:
            result[player_id] = "young"
        elif age <= 31:
            result[player_id] = "prime"
        else:
            result[player_id] = "veteran"
    logger.debug("Assigned %d age cohorts", len(result))
    return result


def assign_experience_cohorts(prior_batting: list[BattingStats], player_ids: set[int]) -> dict[int, str]:
    """Assign players to experience cohorts based on career PA before the evaluation season."""
    career_pa: dict[int, int] = defaultdict(int)
    for stat in prior_batting:
        if stat.player_id in player_ids and stat.pa is not None:
            career_pa[stat.player_id] += stat.pa

    result: dict[int, str] = {}
    for pid in player_ids:
        pa = career_pa.get(pid, 0)
        if pa <= 200:
            result[pid] = "rookie"
        elif pa <= 1000:
            result[pid] = "limited"
        else:
            result[pid] = "established"
    logger.debug("Assigned %d experience cohorts", len(result))
    return result


def assign_top300_cohorts(actuals: list[BattingStats], top_n: int = 300) -> dict[int, str]:
    """Assign players to top-N vs rest cohorts based on WAR ranking."""
    sorted_actuals = sorted(actuals, key=lambda x: x.war if x.war is not None else 0.0, reverse=True)
    top_ids = {a.player_id for a in sorted_actuals[:top_n]}
    result: dict[int, str] = {}
    for a in actuals:
        result[a.player_id] = "top300" if a.player_id in top_ids else "rest"
    logger.debug("Assigned %d top300 cohorts", len(result))
    return result
