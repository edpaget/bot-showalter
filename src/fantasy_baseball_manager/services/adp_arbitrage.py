from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    ArbitrageReport,
    FallingPlayer,
    ReachPick,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoardRow
    from fantasy_baseball_manager.services.draft_state import DraftPick


_CATEGORY_BOOST_WEIGHT = 0.5


def detect_falling_players(
    current_pick: int,
    available: list[DraftBoardRow],
    *,
    threshold: int = 10,
    limit: int = 20,
    category_scores: dict[int, float] | None = None,
) -> list[FallingPlayer]:
    with_adp = [r for r in available if r.adp_overall is not None]

    # Value rank among all available players with ADP data
    by_value = sorted(with_adp, key=lambda r: r.value, reverse=True)
    value_rank_map = {r.player_id: rank for rank, r in enumerate(by_value, 1)}

    falling: list[FallingPlayer] = []
    for row in with_adp:
        adp = row.adp_overall
        if adp is None:  # narrowing for type checker; filtered above
            continue
        picks_past = current_pick - adp
        if picks_past <= threshold:
            continue
        base_score = row.value * math.log(1 + picks_past)
        if category_scores is not None:
            cat_score = category_scores.get(row.player_id, 0.0)
            score = base_score * (1 + _CATEGORY_BOOST_WEIGHT * cat_score)
        else:
            score = base_score
        falling.append(
            FallingPlayer(
                player_id=row.player_id,
                player_name=row.player_name,
                position=row.position,
                adp=adp,
                current_pick=current_pick,
                picks_past_adp=picks_past,
                value=row.value,
                value_rank=value_rank_map[row.player_id],
                arbitrage_score=score,
                player_type=row.player_type,
            )
        )

    falling.sort(key=lambda f: f.arbitrage_score, reverse=True)
    return falling[:limit]


def detect_reaches(
    picks: list[DraftPick],
    adp_lookup: dict[int, float],
    *,
    threshold: int = 10,
) -> list[ReachPick]:
    reaches: list[ReachPick] = []
    for pick in picks:
        adp = adp_lookup.get(pick.player_id)
        if adp is None:
            continue
        ahead = adp - pick.pick_number
        if ahead <= threshold:
            continue
        reaches.append(
            ReachPick(
                player_id=pick.player_id,
                player_name=pick.player_name,
                position=pick.position,
                player_type=pick.player_type,
                adp=adp,
                pick_number=pick.pick_number,
                picks_ahead_of_adp=ahead,
                drafter_team=pick.team,
            )
        )
    reaches.sort(key=lambda r: r.picks_ahead_of_adp, reverse=True)
    return reaches


def build_arbitrage_report(
    current_pick: int,
    available: list[DraftBoardRow],
    picks: list[DraftPick],
    adp_lookup: dict[int, float],
    *,
    threshold: int = 10,
    limit: int = 20,
    category_scores: dict[int, float] | None = None,
) -> ArbitrageReport:
    return ArbitrageReport(
        current_pick=current_pick,
        falling=detect_falling_players(
            current_pick, available, threshold=threshold, limit=limit, category_scores=category_scores
        ),
        reaches=detect_reaches(picks, adp_lookup, threshold=threshold),
    )
