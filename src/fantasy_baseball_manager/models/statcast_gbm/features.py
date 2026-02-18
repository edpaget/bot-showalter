from collections.abc import Sequence

from fantasy_baseball_manager.features import batting, pitching, player
from fantasy_baseball_manager.features.transforms import (
    BATTED_BALL,
    BATTED_BALL_AGAINST,
    BATTED_BALL_INTERACTIONS,
    COMMAND,
    EXPECTED_STATS,
    PITCH_MIX,
    PLATE_DISCIPLINE,
    SPIN_PROFILE,
    SPRAY_ANGLE,
    SPRINT_SPEED_TRANSFORM,
)
from fantasy_baseball_manager.features.types import (
    AnyFeature,
    Feature,
    FeatureSet,
    SpineFilter,
    TransformFeature,
)

_BATTER_LAG_STATS = ("pa", "hr", "h", "doubles", "triples", "bb", "so", "sb")
_BATTER_LAG_RATE_STATS = ("avg", "obp", "slg")
_BATTER_LAG_DERIVED_RATES: tuple[tuple[str, str, str], ...] = (
    ("k_pct", "so", "pa"),
    ("bb_pct", "bb", "pa"),
)
_PITCHER_LAG_STATS = ("ip", "so", "bb", "hr", "era", "fip")


def _batter_lag_features() -> list[Feature]:
    features: list[Feature] = []
    for stat in (*_BATTER_LAG_STATS, *_BATTER_LAG_RATE_STATS):
        features.append(batting.col(stat).lag(1).alias(f"{stat}_1"))
    for name, numerator, denominator in _BATTER_LAG_DERIVED_RATES:
        features.append(batting.col(numerator).lag(1).per(denominator).alias(f"{name}_1"))
    return features


def _pitcher_lag_features() -> list[Feature]:
    features: list[Feature] = []
    for stat in _PITCHER_LAG_STATS:
        features.append(pitching.col(stat).lag(1).alias(f"{stat}_1"))
    return features


def build_batter_feature_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_batter_lag_features())
    features.extend([BATTED_BALL, PLATE_DISCIPLINE, EXPECTED_STATS, SPRAY_ANGLE])
    return FeatureSet(
        name="statcast_gbm_batting",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def _batter_target_features() -> list[Feature]:
    direct = ("avg", "obp", "slg", "woba")
    counting = ("h", "hr", "ab", "so", "sf")
    features: list[Feature] = []
    for stat in direct:
        features.append(batting.col(stat).lag(0).alias(f"target_{stat}"))
    for stat in counting:
        features.append(batting.col(stat).lag(0).alias(f"target_{stat}"))
    return features


def build_batter_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_batter_lag_features())
    features.extend([BATTED_BALL, PLATE_DISCIPLINE, EXPECTED_STATS, SPRAY_ANGLE])
    features.extend(_batter_target_features())
    return FeatureSet(
        name="statcast_gbm_batting_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def batter_feature_columns() -> list[str]:
    fs = build_batter_feature_set([])
    columns: list[str] = []
    for f in fs.features:
        if isinstance(f, TransformFeature):
            columns.extend(f.outputs)
        elif isinstance(f, Feature):
            columns.append(f.name)
    return columns


def build_pitcher_feature_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_pitcher_lag_features())
    features.extend([PITCH_MIX, SPIN_PROFILE, PLATE_DISCIPLINE, BATTED_BALL_AGAINST, COMMAND])
    return FeatureSet(
        name="statcast_gbm_pitching",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )


def _pitcher_target_features() -> list[Feature]:
    direct = ("era", "fip", "k_per_9", "bb_per_9", "whip")
    counting = ("h", "hr", "ip", "so")
    features: list[Feature] = []
    for stat in direct:
        features.append(pitching.col(stat).lag(0).alias(f"target_{stat}"))
    for stat in counting:
        features.append(pitching.col(stat).lag(0).alias(f"target_{stat}"))
    return features


def build_pitcher_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_pitcher_lag_features())
    features.extend([PITCH_MIX, SPIN_PROFILE, PLATE_DISCIPLINE, BATTED_BALL_AGAINST, COMMAND])
    features.extend(_pitcher_target_features())
    return FeatureSet(
        name="statcast_gbm_pitching_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )


def pitcher_feature_columns() -> list[str]:
    fs = build_pitcher_feature_set([])
    columns: list[str] = []
    for f in fs.features:
        if isinstance(f, TransformFeature):
            columns.extend(f.outputs)
        elif isinstance(f, Feature):
            columns.append(f.name)
    return columns


# --- Preseason (lagged Statcast) feature sets ---


def build_batter_preseason_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_batter_lag_features())
    features.extend(
        [
            BATTED_BALL.with_lag(1),
            PLATE_DISCIPLINE.with_lag(1),
            EXPECTED_STATS.with_lag(1),
            SPRAY_ANGLE.with_lag(1),
        ]
    )
    return FeatureSet(
        name="statcast_gbm_batting_preseason",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def build_batter_preseason_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_batter_lag_features())
    features.extend(
        [
            BATTED_BALL.with_lag(1),
            PLATE_DISCIPLINE.with_lag(1),
            EXPECTED_STATS.with_lag(1),
            SPRAY_ANGLE.with_lag(1),
        ]
    )
    features.extend(_batter_target_features())
    return FeatureSet(
        name="statcast_gbm_batting_preseason_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def batter_preseason_feature_columns() -> list[str]:
    fs = build_batter_preseason_set([])
    columns: list[str] = []
    for f in fs.features:
        if isinstance(f, TransformFeature):
            columns.extend(f.outputs)
        elif isinstance(f, Feature):
            columns.append(f.name)
    return columns


def build_pitcher_preseason_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_pitcher_lag_features())
    features.extend(
        [
            PITCH_MIX.with_lag(1),
            SPIN_PROFILE.with_lag(1),
            PLATE_DISCIPLINE.with_lag(1),
            BATTED_BALL_AGAINST.with_lag(1),
            COMMAND.with_lag(1),
        ]
    )
    return FeatureSet(
        name="statcast_gbm_pitching_preseason",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )


def build_pitcher_preseason_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_pitcher_lag_features())
    features.extend(
        [
            PITCH_MIX.with_lag(1),
            SPIN_PROFILE.with_lag(1),
            PLATE_DISCIPLINE.with_lag(1),
            BATTED_BALL_AGAINST.with_lag(1),
            COMMAND.with_lag(1),
        ]
    )
    features.extend(_pitcher_target_features())
    return FeatureSet(
        name="statcast_gbm_pitching_preseason_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )


def pitcher_preseason_feature_columns() -> list[str]:
    fs = build_pitcher_preseason_set([])
    columns: list[str] = []
    for f in fs.features:
        if isinstance(f, TransformFeature):
            columns.extend(f.outputs)
        elif isinstance(f, Feature):
            columns.append(f.name)
    return columns


# --- Weighted preseason (recency-biased blended Statcast) feature sets ---


def build_batter_preseason_weighted_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_batter_lag_features())
    features.extend(
        [
            BATTED_BALL.with_weighted_lag((1, 2), (0.7, 0.3)),
            PLATE_DISCIPLINE.with_weighted_lag((1, 2), (0.7, 0.3)),
            EXPECTED_STATS.with_weighted_lag((1, 2), (0.7, 0.3)),
            SPRAY_ANGLE.with_weighted_lag((1, 2), (0.7, 0.3)),
        ]
    )
    return FeatureSet(
        name="statcast_gbm_batting_preseason_weighted",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def build_batter_preseason_weighted_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_batter_lag_features())
    features.extend(
        [
            BATTED_BALL.with_weighted_lag((1, 2), (0.7, 0.3)),
            PLATE_DISCIPLINE.with_weighted_lag((1, 2), (0.7, 0.3)),
            EXPECTED_STATS.with_weighted_lag((1, 2), (0.7, 0.3)),
            SPRAY_ANGLE.with_weighted_lag((1, 2), (0.7, 0.3)),
        ]
    )
    features.extend(_batter_target_features())
    return FeatureSet(
        name="statcast_gbm_batting_preseason_weighted_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def batter_preseason_weighted_feature_columns() -> list[str]:
    fs = build_batter_preseason_weighted_set([])
    columns: list[str] = []
    for f in fs.features:
        if isinstance(f, TransformFeature):
            columns.extend(f.outputs)
        elif isinstance(f, Feature):
            columns.append(f.name)
    return columns


# --- Averaged preseason (multi-year pooled Statcast) feature sets ---


def build_batter_preseason_averaged_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_batter_lag_features())
    features.extend(
        [
            BATTED_BALL.with_avg_lag(1, 2),
            PLATE_DISCIPLINE.with_avg_lag(1, 2),
            EXPECTED_STATS.with_avg_lag(1, 2),
            SPRAY_ANGLE.with_avg_lag(1, 2),
        ]
    )
    return FeatureSet(
        name="statcast_gbm_batting_preseason_avg",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def build_batter_preseason_averaged_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_batter_lag_features())
    features.extend(
        [
            BATTED_BALL.with_avg_lag(1, 2),
            PLATE_DISCIPLINE.with_avg_lag(1, 2),
            EXPECTED_STATS.with_avg_lag(1, 2),
            SPRAY_ANGLE.with_avg_lag(1, 2),
        ]
    )
    features.extend(_batter_target_features())
    return FeatureSet(
        name="statcast_gbm_batting_preseason_avg_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def batter_preseason_averaged_feature_columns() -> list[str]:
    fs = build_batter_preseason_averaged_set([])
    columns: list[str] = []
    for f in fs.features:
        if isinstance(f, TransformFeature):
            columns.extend(f.outputs)
        elif isinstance(f, Feature):
            columns.append(f.name)
    return columns


def build_pitcher_preseason_averaged_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_pitcher_lag_features())
    features.extend(
        [
            PITCH_MIX.with_avg_lag(1, 2),
            SPIN_PROFILE.with_avg_lag(1, 2),
            PLATE_DISCIPLINE.with_avg_lag(1, 2),
            BATTED_BALL_AGAINST.with_avg_lag(1, 2),
            COMMAND.with_avg_lag(1, 2),
        ]
    )
    return FeatureSet(
        name="statcast_gbm_pitching_preseason_avg",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )


def build_pitcher_preseason_averaged_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_pitcher_lag_features())
    features.extend(
        [
            PITCH_MIX.with_avg_lag(1, 2),
            SPIN_PROFILE.with_avg_lag(1, 2),
            PLATE_DISCIPLINE.with_avg_lag(1, 2),
            BATTED_BALL_AGAINST.with_avg_lag(1, 2),
            COMMAND.with_avg_lag(1, 2),
        ]
    )
    features.extend(_pitcher_target_features())
    return FeatureSet(
        name="statcast_gbm_pitching_preseason_avg_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )


def pitcher_preseason_averaged_feature_columns() -> list[str]:
    fs = build_pitcher_preseason_averaged_set([])
    columns: list[str] = []
    for f in fs.features:
        if isinstance(f, TransformFeature):
            columns.extend(f.outputs)
        elif isinstance(f, Feature):
            columns.append(f.name)
    return columns


# --- Curated (pruned) column lists ---
# Derived from ablation study (2026-02-16). Each list removes features
# with zero or negative permutation importance for that mode.


def live_batter_curated_columns() -> list[str]:
    return [
        "avg_exit_velo",
        "max_exit_velo",
        "avg_launch_angle",
        "barrel_pct",
        "hard_hit_pct",
        "gb_pct",
        "fb_pct",
        "sweet_spot_pct",
        "exit_velo_p90",
        "chase_rate",
        "zone_contact_pct",
        "whiff_rate",
        "swinging_strike_pct",
        "called_strike_pct",
        "xba",
        "xwoba",
        "xslg",
        "pull_pct",
        "oppo_pct",
        "center_pct",
        "sprint_speed",
    ]


def live_pitcher_curated_columns() -> list[str]:
    return [
        "ff_velo",
        "sl_velo",
        "ch_pct",
        "cu_pct",
        "fc_pct",
        "fc_velo",
        "avg_spin_rate",
        "ff_spin",
        "avg_h_break",
        "ff_v_break",
        "sl_h_break",
        "avg_extension",
        "ff_extension",
        "chase_rate",
        "zone_contact_pct",
        "whiff_rate",
        "swinging_strike_pct",
        "called_strike_pct",
        "gb_pct_against",
        "fb_pct_against",
        "avg_exit_velo_against",
        "barrel_pct_against",
        "zone_rate",
    ]


def preseason_batter_curated_columns() -> list[str]:
    return [
        "age",
        "pa_1",
        "doubles_1",
        "bb_1",
        "so_1",
        "avg_1",
        "obp_1",
        "slg_1",
        "k_pct_1",
        "bb_pct_1",
        "avg_exit_velo",
        "max_exit_velo",
        "barrel_pct",
        "gb_pct",
        "fb_pct",
        "ld_pct",
        "exit_velo_p90",
        "zone_contact_pct",
        "swinging_strike_pct",
        "xba",
        "xslg",
        "pull_pct",
        "oppo_pct",
        "center_pct",
    ]


def preseason_pitcher_curated_columns() -> list[str]:
    return [
        "age",
        "ip_1",
        "so_1",
        "bb_1",
        "era_1",
        "fip_1",
        "ff_pct",
        "ff_velo",
        "sl_pct",
        "ch_pct",
        "ch_velo",
        "cu_pct",
        "cu_velo",
        "si_pct",
        "fc_pct",
        "fc_velo",
        "avg_spin_rate",
        "ff_spin",
        "sl_spin",
        "cu_spin",
        "ch_spin",
        "avg_h_break",
        "ff_h_break",
        "ff_v_break",
        "cu_h_break",
        "cu_v_break",
        "ch_h_break",
        "ch_v_break",
        "ff_extension",
        "chase_rate",
        "zone_contact_pct",
        "whiff_rate",
        "called_strike_pct",
        "gb_pct_against",
        "avg_exit_velo_against",
        "barrel_pct_against",
        "zone_rate",
        "first_pitch_strike_pct",
    ]


# --- Curated (pruned) feature-set builders ---
# These build FeatureSets with pruned Feature objects (lag stats).
# TransformFeature objects still produce all their outputs; per-output
# pruning is handled by column lists at training/prediction time.

_LIVE_BATTER_LAG_STATS: tuple[str, ...] = ()
_LIVE_PITCHER_LAG_STATS: tuple[str, ...] = ()
_PRESEASON_BATTER_LAG_STATS = ("pa", "doubles", "bb", "so")
_PRESEASON_BATTER_LAG_RATE_STATS = ("avg", "obp", "slg")
_PRESEASON_PITCHER_LAG_STATS = ("ip", "so", "bb", "era", "fip")


def _curated_batter_lag_features(
    stats: tuple[str, ...],
    rate_stats: tuple[str, ...] = (),
    derived_rates: tuple[tuple[str, str, str], ...] = (),
) -> list[Feature]:
    features: list[Feature] = []
    for stat in (*stats, *rate_stats):
        features.append(batting.col(stat).lag(1).alias(f"{stat}_1"))
    for name, numerator, denominator in derived_rates:
        features.append(batting.col(numerator).lag(1).per(denominator).alias(f"{name}_1"))
    return features


def _curated_pitcher_lag_features(stats: tuple[str, ...]) -> list[Feature]:
    features: list[Feature] = []
    for stat in stats:
        features.append(pitching.col(stat).lag(1).alias(f"{stat}_1"))
    return features


def build_live_batter_feature_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = []
    features.extend(_curated_batter_lag_features(_LIVE_BATTER_LAG_STATS))
    features.extend([BATTED_BALL, PLATE_DISCIPLINE, EXPECTED_STATS, SPRAY_ANGLE])
    features.append(SPRINT_SPEED_TRANSFORM)
    features.append(BATTED_BALL_INTERACTIONS)
    return FeatureSet(
        name="statcast_gbm_batting_live",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def build_live_batter_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = []
    features.extend(_curated_batter_lag_features(_LIVE_BATTER_LAG_STATS))
    features.extend([BATTED_BALL, PLATE_DISCIPLINE, EXPECTED_STATS, SPRAY_ANGLE])
    features.append(SPRINT_SPEED_TRANSFORM)
    features.append(BATTED_BALL_INTERACTIONS)
    features.extend(_batter_target_features())
    return FeatureSet(
        name="statcast_gbm_batting_live_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def build_live_pitcher_feature_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = []
    features.extend(_curated_pitcher_lag_features(_LIVE_PITCHER_LAG_STATS))
    features.extend([PITCH_MIX, SPIN_PROFILE, PLATE_DISCIPLINE, BATTED_BALL_AGAINST, COMMAND])
    return FeatureSet(
        name="statcast_gbm_pitching_live",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )


def build_live_pitcher_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = []
    features.extend(_curated_pitcher_lag_features(_LIVE_PITCHER_LAG_STATS))
    features.extend([PITCH_MIX, SPIN_PROFILE, PLATE_DISCIPLINE, BATTED_BALL_AGAINST, COMMAND])
    features.extend(_pitcher_target_features())
    return FeatureSet(
        name="statcast_gbm_pitching_live_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )


def build_preseason_batter_curated_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(
        _curated_batter_lag_features(
            _PRESEASON_BATTER_LAG_STATS,
            _PRESEASON_BATTER_LAG_RATE_STATS,
            _BATTER_LAG_DERIVED_RATES,
        )
    )
    features.extend(
        [
            BATTED_BALL.with_lag(1),
            PLATE_DISCIPLINE.with_lag(1),
            EXPECTED_STATS.with_lag(1),
            SPRAY_ANGLE.with_lag(1),
        ]
    )
    return FeatureSet(
        name="statcast_gbm_batting_preseason_curated",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def build_preseason_batter_curated_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(
        _curated_batter_lag_features(
            _PRESEASON_BATTER_LAG_STATS,
            _PRESEASON_BATTER_LAG_RATE_STATS,
            _BATTER_LAG_DERIVED_RATES,
        )
    )
    features.extend(
        [
            BATTED_BALL.with_lag(1),
            PLATE_DISCIPLINE.with_lag(1),
            EXPECTED_STATS.with_lag(1),
            SPRAY_ANGLE.with_lag(1),
        ]
    )
    features.extend(_batter_target_features())
    return FeatureSet(
        name="statcast_gbm_batting_preseason_curated_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="batter"),
    )


def build_preseason_pitcher_curated_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_curated_pitcher_lag_features(_PRESEASON_PITCHER_LAG_STATS))
    features.extend(
        [
            PITCH_MIX.with_lag(1),
            SPIN_PROFILE.with_lag(1),
            PLATE_DISCIPLINE.with_lag(1),
            BATTED_BALL_AGAINST.with_lag(1),
            COMMAND.with_lag(1),
        ]
    )
    return FeatureSet(
        name="statcast_gbm_pitching_preseason_curated",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )


def build_preseason_pitcher_curated_training_set(seasons: Sequence[int]) -> FeatureSet:
    features: list[AnyFeature] = [player.age()]
    features.extend(_curated_pitcher_lag_features(_PRESEASON_PITCHER_LAG_STATS))
    features.extend(
        [
            PITCH_MIX.with_lag(1),
            SPIN_PROFILE.with_lag(1),
            PLATE_DISCIPLINE.with_lag(1),
            BATTED_BALL_AGAINST.with_lag(1),
            COMMAND.with_lag(1),
        ]
    )
    features.extend(_pitcher_target_features())
    return FeatureSet(
        name="statcast_gbm_pitching_preseason_curated_train",
        features=tuple(features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
        spine_filter=SpineFilter(player_type="pitcher"),
    )
