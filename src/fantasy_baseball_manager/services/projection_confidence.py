import statistics

from fantasy_baseball_manager.domain.league_settings import LeagueSettings, StatType
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.projection_confidence import (
    ConfidenceReport,
    PlayerConfidence,
    StatSpread,
)


def compute_confidence(
    projections: list[Projection],
    league: LeagueSettings,
    player_names: dict[int, str],
    *,
    min_systems: int = 3,
    positions: dict[int, str] | None = None,
) -> ConfidenceReport:
    """Compute cross-system projection confidence for each player.

    Groups projections by (player_id, player_type), deduplicates by system
    (first version wins), and computes per-stat spread metrics for league-
    relevant categories. Players with fewer than ``min_systems`` distinct
    systems are excluded.

    ``overall_cv`` is the mean CV across counting stats only (rate stat CVs
    are misleading). Agreement thresholds: <= 0.10 high, >= 0.25 low, else
    medium.
    """
    if not projections:
        return ConfidenceReport(season=0, systems=[], players=[])

    season = projections[0].season

    # Build category lookups per player_type
    batting_cats = league.batting_categories
    pitching_cats = league.pitching_categories

    # Group projections by (player_id, player_type)
    grouped: dict[tuple[int, str], list[Projection]] = {}
    all_systems: set[str] = set()
    for proj in grouped_projections(projections):
        key = (proj.player_id, proj.player_type)
        grouped.setdefault(key, []).append(proj)
        all_systems.add(proj.system)

    players: list[PlayerConfidence] = []
    for (player_id, player_type), player_projs in grouped.items():
        # Dedup: one projection per system (first encountered wins)
        seen_systems: set[str] = set()
        deduped: list[Projection] = []
        for proj in player_projs:
            if proj.system not in seen_systems:
                seen_systems.add(proj.system)
                deduped.append(proj)

        if len(deduped) < min_systems:
            continue

        categories = pitching_cats if player_type == "pitcher" else batting_cats

        spreads: list[StatSpread] = []
        counting_cvs: list[float] = []

        for cat in categories:
            systems_values: dict[str, float] = {}
            for proj in deduped:
                val = proj.stat_json.get(cat.key)
                if val is not None:
                    systems_values[proj.system] = float(val)

            if len(systems_values) < min_systems:
                continue

            values = list(systems_values.values())
            mean = statistics.mean(values)
            std = statistics.pstdev(values)
            cv = std / mean if mean != 0 else 0.0

            spread = StatSpread(
                stat=cat.key,
                min_value=min(values),
                max_value=max(values),
                mean=mean,
                std=std,
                cv=cv,
                systems=systems_values,
            )
            spreads.append(spread)

            if cat.stat_type == StatType.COUNTING:
                counting_cvs.append(cv)

        overall_cv = statistics.mean(counting_cvs) if counting_cvs else 0.0
        agreement_level = _classify_agreement(overall_cv)

        position = (positions or {}).get(player_id, "")
        player_name = player_names.get(player_id, str(player_id))

        players.append(
            PlayerConfidence(
                player_id=player_id,
                player_name=player_name,
                player_type=player_type,
                position=position,
                spreads=spreads,
                overall_cv=overall_cv,
                agreement_level=agreement_level,
            )
        )

    # Sort by overall_cv descending (most uncertain first)
    players.sort(key=lambda p: p.overall_cv, reverse=True)

    return ConfidenceReport(
        season=season,
        systems=sorted(all_systems),
        players=players,
    )


def grouped_projections(projections: list[Projection]) -> list[Projection]:
    """Return projections sorted by (player_id, player_type, system) for grouping."""
    return sorted(projections, key=lambda p: (p.player_id, p.player_type, p.system))


def _classify_agreement(overall_cv: float) -> str:
    if overall_cv <= 0.10:
        return "high"
    if overall_cv >= 0.25:
        return "low"
    return "medium"
