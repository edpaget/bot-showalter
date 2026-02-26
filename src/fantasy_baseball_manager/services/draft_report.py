from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.draft_report import (
    CategoryStanding,
    DraftReport,
    PickGrade,
    StealOrReach,
)
from fantasy_baseball_manager.services.draft_state import DraftFormat, DraftState

# Flex slot mappings: batter positions can flex to UTIL, pitcher to P
_BATTER_FLEX = "UTIL"
_PITCHER_FLEX = "P"
_BATTER_TYPE = "batter"
_PITCHER_TYPE = "pitcher"


def _can_fill_slot(player: DraftBoardRow, slot: str) -> bool:
    """Check if a player can fill a given roster slot."""
    if player.position == slot:
        return True
    if slot == _BATTER_FLEX and player.player_type == _BATTER_TYPE:
        return True
    return slot == _PITCHER_FLEX and player.player_type == _PITCHER_TYPE


def _compute_optimal_value(
    full_pool: list[DraftBoardRow],
    roster_slots: dict[str, int],
) -> float:
    """Compute optimal total value via greedy assignment."""
    sorted_players = sorted(full_pool, key=lambda p: p.value, reverse=True)
    remaining: dict[str, int] = dict(roster_slots)
    total = 0.0

    for player in sorted_players:
        # Try primary position first
        if player.position in remaining and remaining[player.position] > 0:
            remaining[player.position] -= 1
            total += player.value
            continue
        # Try flex slot
        flex = _BATTER_FLEX if player.player_type == _BATTER_TYPE else _PITCHER_FLEX
        if flex in remaining and remaining[flex] > 0:
            remaining[flex] -= 1
            total += player.value
            continue

    return total


def _compute_category_standings(
    state: DraftState,
    pool_lookup: dict[int, DraftBoardRow],
    categories: tuple[str, ...],
) -> list[CategoryStanding]:
    """Compute per-category standings across all teams."""
    teams = state.config.teams
    standings: list[CategoryStanding] = []

    for cat in categories:
        team_totals: dict[int, float] = {}
        for team_num in range(1, teams + 1):
            total_z = 0.0
            for pick in state.team_rosters[team_num]:
                row = pool_lookup.get(pick.player_id)
                if row is not None:
                    total_z += row.category_z_scores.get(cat, 0.0)
            team_totals[team_num] = total_z

        # Rank all teams (higher z = better rank = lower number)
        sorted_teams = sorted(team_totals.items(), key=lambda x: x[1], reverse=True)
        user_team = state.config.user_team
        user_rank = next(i + 1 for i, (t, _) in enumerate(sorted_teams) if t == user_team)
        user_z = team_totals[user_team]

        standings.append(
            CategoryStanding(
                category=cat,
                total_z=user_z,
                rank=user_rank,
                teams=teams,
            )
        )

    return standings


def _best_available_for_slot(
    slot: str,
    available: dict[int, DraftBoardRow],
) -> float:
    """Find the best available player value that can fill a slot."""
    best = 0.0
    for player in available.values():
        if _can_fill_slot(player, slot) and player.value > best:
            best = player.value
    return best


def _compute_pick_grades(
    state: DraftState,
    full_pool: list[DraftBoardRow],
) -> list[PickGrade]:
    """Grade each user pick by comparing to best available at pick time."""
    pool_lookup = {p.player_id: p for p in full_pool}
    # Replay the draft: start with full pool, remove players as they're picked
    available: dict[int, DraftBoardRow] = {p.player_id: p for p in full_pool}
    user_team = state.config.user_team
    grades: list[PickGrade] = []

    for pick in state.picks:
        if pick.team == user_team:
            best_value = _best_available_for_slot(pick.position, available)
            row = pool_lookup[pick.player_id]
            grade = row.value / best_value if best_value > 0 else 1.0
            grades.append(
                PickGrade(
                    pick_number=pick.pick_number,
                    player_id=pick.player_id,
                    player_name=pick.player_name,
                    position=pick.position,
                    value=row.value,
                    best_available_value=best_value,
                    grade=grade,
                )
            )
        # Remove picked player from available pool
        available.pop(pick.player_id, None)

    return grades


def _compute_steals_reaches(
    pick_grades: list[PickGrade],
    pool_lookup: dict[int, DraftBoardRow],
    threshold: int,
) -> tuple[list[StealOrReach], list[StealOrReach]]:
    """Identify steals and reaches based on ADP vs pick position."""
    steals: list[StealOrReach] = []
    reaches: list[StealOrReach] = []

    for grade in pick_grades:
        row = pool_lookup.get(grade.player_id)
        if row is None:
            continue

        # Use adp_rank if available, otherwise fall back to rank (value rank)
        expected_pick = row.adp_rank if row.adp_rank is not None else row.rank
        pick_delta = expected_pick - grade.pick_number

        if abs(pick_delta) <= threshold:
            continue

        entry = StealOrReach(
            pick_number=grade.pick_number,
            player_id=grade.player_id,
            player_name=grade.player_name,
            position=grade.position,
            value=grade.value,
            pick_delta=pick_delta,
        )

        if pick_delta > 0:
            steals.append(entry)
        else:
            reaches.append(entry)

    return steals, reaches


def draft_report(
    state: DraftState,
    full_pool: list[DraftBoardRow],
    *,
    batting_categories: tuple[str, ...] = (),
    pitching_categories: tuple[str, ...] = (),
    steal_threshold: int = 5,
) -> DraftReport:
    """Generate a post-draft analysis report."""
    pool_lookup = {p.player_id: p for p in full_pool}
    user_team = state.config.user_team
    config = state.config

    # Total value of user's picks
    user_picks = state.team_rosters.get(user_team, [])
    total_value = sum(pool_lookup[p.player_id].value for p in user_picks if p.player_id in pool_lookup)

    # Optimal value
    optimal_value = _compute_optimal_value(full_pool, config.roster_slots)

    # Value efficiency
    value_efficiency = total_value / optimal_value if optimal_value > 0 else 0.0

    # Category standings
    all_categories = batting_categories + pitching_categories
    category_standings = _compute_category_standings(state, pool_lookup, all_categories)

    # Pick grades
    pick_grades = _compute_pick_grades(state, full_pool)
    mean_grade = sum(g.grade for g in pick_grades) / len(pick_grades) if pick_grades else 0.0

    # Steals and reaches
    steals, reaches = _compute_steals_reaches(pick_grades, pool_lookup, steal_threshold)

    # Budget info (auction only)
    budget: int | None = None
    total_spent: int | None = None
    if config.format == DraftFormat.AUCTION:
        budget = config.budget
        total_spent = sum(p.price for p in user_picks if p.price is not None)

    return DraftReport(
        total_value=total_value,
        optimal_value=optimal_value,
        value_efficiency=value_efficiency,
        budget=budget,
        total_spent=total_spent,
        category_standings=category_standings,
        pick_grades=pick_grades,
        mean_grade=mean_grade,
        steals=steals,
        reaches=reaches,
    )
