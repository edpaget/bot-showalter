import statistics

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.league_settings import LeagueSettings, StatType
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.projection_confidence import (
    ClassifiedPlayer,
    ConfidenceReport,
    PlayerConfidence,
    StatSpread,
    VarianceClassification,
)
from fantasy_baseball_manager.domain.valuation import Valuation


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


# ---------------------------------------------------------------------------
# Phase 2: High-variance player classification
# ---------------------------------------------------------------------------

_ADP_CLOSE_THRESHOLD: int = 10
_ADP_LATE_THRESHOLD: int = 20
_ADP_EARLY_THRESHOLD: int = 20
_SIGNIFICANT_ABOVE_RATIO: float = 1.3


def classify_variance(
    report: ConfidenceReport,
    valuations: list[Valuation],
    adp: list[ADP] | None = None,
) -> list[ClassifiedPlayer]:
    """Classify players into draft-actionable variance buckets.

    Combines projection agreement (from *report*) with dollar valuations and
    ADP to produce a single ``VarianceClassification`` per player.
    """
    # 1. Group valuations by player_id → per-player value stats
    player_values: dict[int, list[float]] = {}
    for v in valuations:
        player_values.setdefault(v.player_id, []).append(v.value)

    # 2. Compute median, max, min per player
    player_stats: dict[int, tuple[float, float, float]] = {}  # median, max, min
    for pid, vals in player_values.items():
        player_stats[pid] = (
            statistics.median(vals),
            max(vals),
            min(vals),
        )

    # 3. Compute value_rank — sort by median desc, assign ranks 1..N
    ranked_pids: list[int] = sorted(player_stats, key=lambda pid: player_stats[pid][0], reverse=True)
    value_rank_map: dict[int, int] = {pid: rank for rank, pid in enumerate(ranked_pids, start=1)}

    # 4. Build rank→value mapping
    rank_to_value: dict[int, float] = {rank: player_stats[pid][0] for pid, rank in value_rank_map.items()}
    max_rank: int = len(rank_to_value)

    # 5. Build ADP lookup — group by player_id, take lowest overall_pick
    adp_lookup: dict[int, float] = {}
    if adp:
        for entry in adp:
            current = adp_lookup.get(entry.player_id)
            if current is None or entry.overall_pick < current:
                adp_lookup[entry.player_id] = entry.overall_pick

    # 6. Classify each player in the report
    result: list[ClassifiedPlayer] = []
    for player in report.players:
        if player.player_id not in player_stats:
            continue

        median_val, max_val, min_val = player_stats[player.player_id]
        v_rank: int = value_rank_map[player.player_id]

        # ADP
        raw_adp: float | None = adp_lookup.get(player.player_id)
        adp_rank: int | None = round(raw_adp) if raw_adp is not None else None

        # adp_expected: dollar value at the ADP position (or median if no ADP)
        if adp_rank is not None and max_rank > 0:
            clamped: int = max(1, min(adp_rank, max_rank))
            adp_expected: float = rank_to_value[clamped]
        else:
            adp_expected = median_val

        # risk_reward_score
        rr_score: float = max_val + min_val - 2 * adp_expected

        # Upside flag
        has_upside: bool = max_val >= _SIGNIFICANT_ABOVE_RATIO * adp_expected

        # Classification decision tree
        classification = _classify_player(player.agreement_level, adp_rank, v_rank, has_upside)

        result.append(
            ClassifiedPlayer(
                player=player,
                classification=classification,
                adp_rank=adp_rank,
                value_rank=v_rank,
                risk_reward_score=rr_score,
            )
        )

    return result


def _classify_player(
    agreement: str,
    adp_rank: int | None,
    value_rank: int,
    has_upside: bool,
) -> VarianceClassification:
    """Decision tree: first match wins → exactly one classification."""
    if adp_rank is None:
        # No ADP path
        if agreement in ("high", "medium"):
            return VarianceClassification.KNOWN_QUANTITY
        return VarianceClassification.HIDDEN_UPSIDE

    rank_diff: int = adp_rank - value_rank
    adp_late: bool = rank_diff > _ADP_LATE_THRESHOLD
    adp_early: bool = rank_diff < -_ADP_EARLY_THRESHOLD

    # Has ADP path
    if agreement == "high" and abs(rank_diff) <= _ADP_CLOSE_THRESHOLD:
        return VarianceClassification.SAFE_CONSENSUS
    if agreement == "high":
        return VarianceClassification.KNOWN_QUANTITY
    if agreement == "low" and adp_late and has_upside:
        return VarianceClassification.UPSIDE_GAMBLE
    if agreement == "low" and adp_early:
        return VarianceClassification.RISKY_AVOID
    if agreement == "medium" and has_upside:
        return VarianceClassification.HIDDEN_UPSIDE
    if agreement == "low" and has_upside:
        return VarianceClassification.UPSIDE_GAMBLE
    if agreement == "low":
        return VarianceClassification.RISKY_AVOID
    # medium fallback
    return VarianceClassification.KNOWN_QUANTITY
