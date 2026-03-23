import math
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import (
    ADP,
    BreakoutPrediction,
    ClassifierCalibrationBin,
    ClassifierEvaluation,
    LabelConfig,
    LabeledSeason,
    LiftResult,
    OutcomeLabel,
    ThresholdMetrics,
    Valuation,
)

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

    # Valuation lookup keyed by (player_id, player_type) to avoid collisions
    # for two-way players. Keep best (highest value) per type.
    val_map: dict[tuple[int, str], Valuation] = {}
    for v in valuations:
        key = (v.player_id, v.player_type)
        existing = val_map.get(key)
        if existing is None or v.value > existing.value:
            val_map[key] = v

    results: list[LabeledSeason] = []
    for pid, a in adp_map.items():
        # Find best valuation across all types for this player
        v: Valuation | None = None
        for (vid, _), candidate in val_map.items():
            if vid == pid and (v is None or candidate.value > v.value):
                v = candidate
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


_DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
_DEFAULT_TOP_NS = [10, 20, 50]
_LOG_LOSS_EPS = 1e-15


def evaluate_classifier(
    labels: list[LabeledSeason],
    predictions: list[BreakoutPrediction],
    thresholds: list[float] | None = None,
    top_ns: list[int] | None = None,
) -> ClassifierEvaluation:
    """Evaluate breakout/bust classifier predictions against actual labels."""
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS
    if top_ns is None:
        top_ns = _DEFAULT_TOP_NS

    # Match predictions to labels on player_id
    label_map: dict[int, LabeledSeason] = {ls.player_id: ls for ls in labels}
    pred_map: dict[int, BreakoutPrediction] = {p.player_id: p for p in predictions}

    matched_ids = set(label_map.keys()) & set(pred_map.keys())
    if not matched_ids:
        return ClassifierEvaluation(
            threshold_metrics=[],
            calibration_bins=[],
            lift_results=[],
            log_loss=0.0,
            base_rate_log_loss=0.0,
            n_evaluated=0,
        )

    matched_labels = [label_map[pid] for pid in sorted(matched_ids)]
    matched_preds = [pred_map[pid] for pid in sorted(matched_ids)]
    n = len(matched_labels)

    # --- Threshold metrics ---
    threshold_metrics = _compute_threshold_metrics(matched_labels, matched_preds, thresholds)

    # --- Calibration bins ---
    calibration_bins = _compute_calibration_bins(matched_labels, matched_preds)

    # --- Lift ---
    lift_results = _compute_lift(matched_labels, matched_preds, top_ns)

    # --- Log-loss ---
    log_loss_val = _compute_log_loss(matched_labels, matched_preds)
    base_rate_log_loss_val = _compute_base_rate_log_loss(matched_labels)

    return ClassifierEvaluation(
        threshold_metrics=threshold_metrics,
        calibration_bins=calibration_bins,
        lift_results=lift_results,
        log_loss=log_loss_val,
        base_rate_log_loss=base_rate_log_loss_val,
        n_evaluated=n,
    )


def _compute_threshold_metrics(
    labels: list[LabeledSeason],
    preds: list[BreakoutPrediction],
    thresholds: list[float],
) -> list[ThresholdMetrics]:
    results: list[ThresholdMetrics] = []
    for class_label, prob_getter in [
        ("breakout", lambda p: p.p_breakout),
        ("bust", lambda p: p.p_bust),
    ]:
        actual_positive = sum(1 for ls in labels if ls.label.value == class_label)
        for thresh in thresholds:
            flagged_indices = [i for i, p in enumerate(preds) if prob_getter(p) >= thresh]
            flagged = len(flagged_indices)
            tp = sum(1 for i in flagged_indices if labels[i].label.value == class_label)

            precision = tp / flagged if flagged > 0 else 0.0
            recall = tp / actual_positive if actual_positive > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append(
                ThresholdMetrics(
                    label=class_label,
                    threshold=thresh,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    flagged=flagged,
                    true_positives=tp,
                )
            )
    return results


def _compute_calibration_bins(
    labels: list[LabeledSeason],
    preds: list[BreakoutPrediction],
) -> list[ClassifierCalibrationBin]:
    """Compute 10 equal-width bins for breakout and bust probabilities."""
    results: list[ClassifierCalibrationBin] = []
    bin_edges = [(i / 10, (i + 1) / 10) for i in range(10)]

    for class_label, prob_getter in [
        ("breakout", lambda p: p.p_breakout),
        ("bust", lambda p: p.p_bust),
    ]:
        for low, high in bin_edges:
            center = (low + high) / 2
            in_bin_probs: list[float] = []
            in_bin_actuals: list[int] = []
            for i, p in enumerate(preds):
                prob = prob_getter(p)
                if low <= prob < high or (high == 1.0 and prob == 1.0):
                    in_bin_probs.append(prob)
                    in_bin_actuals.append(1 if labels[i].label.value == class_label else 0)

            if in_bin_probs:
                results.append(
                    ClassifierCalibrationBin(
                        bin_center=center,
                        mean_predicted=sum(in_bin_probs) / len(in_bin_probs),
                        mean_actual=sum(in_bin_actuals) / len(in_bin_actuals),
                        count=len(in_bin_probs),
                    )
                )
    return results


def _compute_lift(
    labels: list[LabeledSeason],
    preds: list[BreakoutPrediction],
    top_ns: list[int],
) -> list[LiftResult]:
    n = len(labels)
    results: list[LiftResult] = []

    for class_label, prob_getter in [
        ("breakout", lambda p: p.p_breakout),
        ("bust", lambda p: p.p_bust),
    ]:
        actual_count = sum(1 for ls in labels if ls.label.value == class_label)
        base_rate = actual_count / n if n > 0 else 0.0

        # Sort by probability descending, get indices
        sorted_indices = sorted(range(n), key=lambda i: prob_getter(preds[i]), reverse=True)

        for top_n in top_ns:
            effective_n = min(top_n, n)
            top_indices = sorted_indices[:effective_n]
            top_actual = sum(1 for i in top_indices if labels[i].label.value == class_label)
            flagged_rate = top_actual / effective_n if effective_n > 0 else 0.0
            lift = flagged_rate / base_rate if base_rate > 0 else 0.0

            results.append(
                LiftResult(
                    label=class_label,
                    top_n=top_n,
                    flagged_rate=flagged_rate,
                    base_rate=base_rate,
                    lift=lift,
                )
            )
    return results


def _compute_log_loss(
    labels: list[LabeledSeason],
    preds: list[BreakoutPrediction],
) -> float:
    """Compute multi-class log-loss."""
    total = 0.0
    for ls, p in zip(labels, preds, strict=True):
        probs = {
            OutcomeLabel.BREAKOUT: p.p_breakout,
            OutcomeLabel.BUST: p.p_bust,
            OutcomeLabel.NEUTRAL: p.p_neutral,
        }
        true_prob = max(probs[ls.label], _LOG_LOSS_EPS)
        total -= math.log(true_prob)
    return total / len(labels)


def _compute_base_rate_log_loss(labels: list[LabeledSeason]) -> float:
    """Log-loss of a classifier that always predicts class frequencies."""
    n = len(labels)
    counts: dict[OutcomeLabel, int] = {
        OutcomeLabel.BREAKOUT: 0,
        OutcomeLabel.BUST: 0,
        OutcomeLabel.NEUTRAL: 0,
    }
    for ls in labels:
        counts[ls.label] += 1

    total = 0.0
    for ls in labels:
        freq = max(counts[ls.label] / n, _LOG_LOSS_EPS)
        total -= math.log(freq)
    return total / n


def historical_backtest(
    per_season_labels: dict[int, list[LabeledSeason]],
    per_season_predictions: dict[int, list[BreakoutPrediction]],
    thresholds: list[float] | None = None,
    top_ns: list[int] | None = None,
) -> dict[int, ClassifierEvaluation]:
    """Evaluate classifier per season. Only evaluates seasons with both labels and predictions."""
    results: dict[int, ClassifierEvaluation] = {}
    for season in per_season_labels:
        if season not in per_season_predictions:
            continue
        results[season] = evaluate_classifier(
            per_season_labels[season],
            per_season_predictions[season],
            thresholds=thresholds,
            top_ns=top_ns,
        )
    return results


def find_actionability_threshold(
    evaluation: ClassifierEvaluation,
    label: str,
    min_precision: float = 0.4,
) -> float | None:
    """Find lowest threshold where precision >= min_precision for the given label."""
    candidates = [
        m for m in evaluation.threshold_metrics if m.label == label and m.precision >= min_precision and m.flagged > 0
    ]
    if not candidates:
        return None
    return min(c.threshold for c in candidates)
