from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow, TierAssignment
from fantasy_baseball_manager.domain.league_settings import LeagueSettings
from fantasy_baseball_manager.domain.valuation import Valuation

_PITCHER_POSITIONS = {"SP", "RP"}


def _is_pitcher_adp(adp: ADP) -> bool:
    return all(p.strip() in _PITCHER_POSITIONS for p in adp.positions.split(",") if p.strip())


def _resolve_adp(entries: list[ADP], is_pitcher: bool) -> ADP:
    matching = [e for e in entries if _is_pitcher_adp(e) == is_pitcher]
    return min(matching or entries, key=lambda a: a.overall_pick)


def build_draft_board(
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str],
    *,
    tiers: list[TierAssignment] | None = None,
    adp: list[ADP] | None = None,
) -> DraftBoard:
    batting_categories = tuple(c.key for c in league.batting_categories)
    pitching_categories = tuple(c.key for c in league.pitching_categories)

    tier_lookup: dict[int, int] = {}
    if tiers is not None:
        tier_lookup = {t.player_id: t.tier for t in tiers}

    adp_by_player: dict[int, list[ADP]] = {}
    if adp is not None:
        for entry in adp:
            adp_by_player.setdefault(entry.player_id, []).append(entry)

    sorted_valuations = sorted(valuations, key=lambda v: v.value, reverse=True)

    rows: list[DraftBoardRow] = []
    for rank, val in enumerate(sorted_valuations, start=1):
        is_pitcher = val.player_type == "pitcher"
        cat_keys = pitching_categories if is_pitcher else batting_categories
        category_z_scores = {k: v for k, v in val.category_scores.items() if k in cat_keys}

        tier = tier_lookup.get(val.player_id)

        adp_overall: float | None = None
        adp_rank: int | None = None
        adp_delta: int | None = None
        if val.player_id in adp_by_player:
            best = _resolve_adp(adp_by_player[val.player_id], is_pitcher)
            adp_overall = best.overall_pick
            adp_rank = best.rank
            adp_delta = best.rank - rank

        player_name = player_names.get(val.player_id, f"Unknown ({val.player_id})")

        rows.append(
            DraftBoardRow(
                player_id=val.player_id,
                player_name=player_name,
                rank=rank,
                player_type=val.player_type,
                position=val.position,
                value=val.value,
                category_z_scores=category_z_scores,
                tier=tier,
                adp_overall=adp_overall,
                adp_rank=adp_rank,
                adp_delta=adp_delta,
            )
        )

    return DraftBoard(
        rows=rows,
        batting_categories=batting_categories,
        pitching_categories=pitching_categories,
    )
