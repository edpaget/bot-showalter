from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import ADP, LabelConfig, LabeledSeason, OutcomeLabel, Valuation

if TYPE_CHECKING:
    from fantasy_baseball_manager.features import DatasetAssembler, FeatureSet


def generate_labels(
    adp: list[ADP],
    valuations: list[Valuation],
    config: LabelConfig,
) -> list[LabeledSeason]:
    """Generate breakout/bust/neutral labels by comparing ADP rank to actual value rank."""
    if not adp or not valuations:
        return []

    # Dedup ADP: group by player_id, keep entry with lowest overall_pick
    adp_by_player: dict[int, list[ADP]] = {}
    for a in adp:
        adp_by_player.setdefault(a.player_id, []).append(a)

    adp_map: dict[int, ADP] = {}
    for pid, entries in adp_by_player.items():
        best = min(entries, key=lambda a: a.overall_pick)
        if best.rank <= config.min_adp_rank:
            adp_map[pid] = best

    # Valuation lookup: first match per player
    val_map: dict[int, Valuation] = {}
    for v in valuations:
        if v.player_id not in val_map:
            val_map[v.player_id] = v

    results: list[LabeledSeason] = []
    for pid, a in adp_map.items():
        v = val_map.get(pid)
        if v is None:
            continue

        rank_delta = a.rank - v.rank
        if rank_delta >= config.breakout_threshold:
            label = OutcomeLabel.BREAKOUT
        elif rank_delta <= config.bust_threshold:
            label = OutcomeLabel.BUST
        else:
            label = OutcomeLabel.NEUTRAL

        results.append(
            LabeledSeason(
                player_id=pid,
                season=a.season,
                player_type=v.player_type,
                adp_rank=a.rank,
                adp_pick=a.overall_pick,
                actual_value_rank=v.rank,
                rank_delta=rank_delta,
                label=label,
            )
        )

    return results


def assemble_labeled_dataset(
    labels: list[LabeledSeason],
    assembler: DatasetAssembler,
    feature_set: FeatureSet,
) -> list[dict[str, Any]]:
    """Join labels with pre-season features into an enriched dataset."""
    if not labels:
        return []

    handle = assembler.get_or_materialize(feature_set)
    rows = assembler.read(handle)

    label_lookup: dict[tuple[int, int], LabeledSeason] = {(ls.player_id, ls.season): ls for ls in labels}

    result: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("player_id"), row.get("season"))
        ls = label_lookup.get(key)  # type: ignore[arg-type]
        if ls is None:
            continue

        enriched = dict(row)
        enriched["label"] = ls.label.value
        enriched["rank_delta"] = ls.rank_delta
        enriched["adp_rank"] = ls.adp_rank
        enriched["adp_pick"] = ls.adp_pick
        result.append(enriched)

    return result


def label_distribution(labels: list[LabeledSeason]) -> dict[str, int]:
    """Count labels by outcome class."""
    counts: dict[str, int] = {
        OutcomeLabel.BREAKOUT.value: 0,
        OutcomeLabel.BUST.value: 0,
        OutcomeLabel.NEUTRAL.value: 0,
    }
    for ls in labels:
        counts[ls.label.value] += 1
    return counts
