from fantasy_baseball_manager.features.types import Feature, FeatureSet, SpineFilter, TransformFeature
from fantasy_baseball_manager.models.statcast_gbm.features import (
    batter_feature_columns,
    batter_preseason_feature_columns,
    build_batter_feature_set,
    build_batter_preseason_set,
    build_batter_preseason_training_set,
    build_batter_training_set,
    build_live_batter_feature_set,
    build_live_batter_training_set,
    build_live_pitcher_feature_set,
    build_live_pitcher_training_set,
    build_pitcher_feature_set,
    build_pitcher_preseason_set,
    build_pitcher_preseason_training_set,
    build_pitcher_training_set,
    build_preseason_batter_curated_set,
    build_preseason_batter_curated_training_set,
    build_preseason_pitcher_curated_set,
    build_preseason_pitcher_curated_training_set,
    live_batter_curated_columns,
    live_pitcher_curated_columns,
    pitcher_feature_columns,
    pitcher_preseason_feature_columns,
    preseason_batter_curated_columns,
    preseason_pitcher_curated_columns,
)


class TestBatterFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_batter_feature_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_has_source_filter(self) -> None:
        fs = build_batter_feature_set([2023])
        assert fs.source_filter == "fangraphs"

    def test_has_batter_spine_filter(self) -> None:
        fs = build_batter_feature_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="batter")

    def test_includes_age(self) -> None:
        fs = build_batter_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_includes_statcast_transforms(self) -> None:
        fs = build_batter_feature_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "batted_ball" in transform_names
        assert "plate_discipline" in transform_names
        assert "expected_stats" in transform_names
        assert "spray_angle" in transform_names

    def test_includes_batting_lags(self) -> None:
        fs = build_batter_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "pa_1" in names
        assert "hr_1" in names

    def test_includes_batting_rate_lags(self) -> None:
        fs = build_batter_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "avg_1" in names
        assert "obp_1" in names
        assert "slg_1" in names
        assert "k_pct_1" in names
        assert "bb_pct_1" in names


class TestBatterTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_batter_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_avg" in names
        assert "target_obp" in names
        assert "target_slg" in names
        assert "target_woba" in names
        assert "target_h" in names
        assert "target_hr" in names

    def test_includes_input_features(self) -> None:
        fs = build_batter_training_set([2023])
        names: list[str] = []
        for f in fs.features:
            if isinstance(f, TransformFeature):
                names.append(f.name)
            elif isinstance(f, Feature):
                names.append(f.name)
        assert "age" in names
        assert "pa_1" in names
        assert "batted_ball" in names

    def test_name(self) -> None:
        fs = build_batter_training_set([2023])
        assert fs.name == "statcast_gbm_batting_train"


class TestBatterFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = batter_feature_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_no_target_columns(self) -> None:
        columns = batter_feature_columns()
        assert not any(c.startswith("target_") for c in columns)

    def test_includes_feature_names(self) -> None:
        columns = batter_feature_columns()
        assert "age" in columns
        assert "pa_1" in columns
        # Rate lag features
        assert "avg_1" in columns
        assert "obp_1" in columns
        assert "slg_1" in columns
        assert "k_pct_1" in columns
        assert "bb_pct_1" in columns
        # TransformFeature outputs should be expanded
        assert "avg_exit_velo" in columns
        assert "chase_rate" in columns
        assert "xba" in columns
        assert "pull_pct" in columns


class TestPitcherFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_pitcher_feature_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_has_source_filter(self) -> None:
        fs = build_pitcher_feature_set([2023])
        assert fs.source_filter == "fangraphs"

    def test_has_pitcher_spine_filter(self) -> None:
        fs = build_pitcher_feature_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="pitcher")

    def test_includes_age(self) -> None:
        fs = build_pitcher_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_includes_statcast_transforms(self) -> None:
        fs = build_pitcher_feature_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "pitch_mix" in transform_names
        assert "spin_profile" in transform_names
        assert "plate_discipline" in transform_names
        assert "batted_ball_against" in transform_names
        assert "command" in transform_names

    def test_includes_pitching_lags(self) -> None:
        fs = build_pitcher_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "ip_1" in names
        assert "so_1" in names


class TestPitcherTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_pitcher_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_era" in names
        assert "target_fip" in names
        assert "target_k_per_9" in names
        assert "target_bb_per_9" in names
        assert "target_whip" in names
        # Counting stats for derived targets
        assert "target_h" in names
        assert "target_hr" in names
        assert "target_ip" in names
        assert "target_so" in names

    def test_includes_input_features(self) -> None:
        fs = build_pitcher_training_set([2023])
        names: list[str] = []
        for f in fs.features:
            if isinstance(f, TransformFeature):
                names.append(f.name)
            elif isinstance(f, Feature):
                names.append(f.name)
        assert "age" in names
        assert "ip_1" in names
        assert "pitch_mix" in names

    def test_name(self) -> None:
        fs = build_pitcher_training_set([2023])
        assert fs.name == "statcast_gbm_pitching_train"


class TestPitcherFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = pitcher_feature_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_no_target_columns(self) -> None:
        columns = pitcher_feature_columns()
        assert not any(c.startswith("target_") for c in columns)

    def test_includes_feature_names(self) -> None:
        columns = pitcher_feature_columns()
        assert "age" in columns
        assert "ip_1" in columns
        # TransformFeature outputs should be expanded
        assert "avg_spin_rate" in columns
        assert "ff_pct" in columns
        assert "chase_rate" in columns
        # Command outputs
        assert "zone_rate" in columns
        # Batted-ball-against outputs
        assert "gb_pct_against" in columns
        assert "barrel_pct_against" in columns
        assert "avg_extension" in columns


class TestBatterPreseasonSet:
    def test_returns_feature_set(self) -> None:
        fs = build_batter_preseason_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_name(self) -> None:
        fs = build_batter_preseason_set([2023])
        assert fs.name == "statcast_gbm_batting_preseason"

    def test_has_batter_spine_filter(self) -> None:
        fs = build_batter_preseason_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="batter")

    def test_includes_age(self) -> None:
        fs = build_batter_preseason_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_includes_batting_lags(self) -> None:
        fs = build_batter_preseason_set([2023])
        names = [f.name for f in fs.features]
        assert "pa_1" in names
        assert "avg_1" in names
        assert "k_pct_1" in names

    def test_statcast_transforms_are_lagged(self) -> None:
        fs = build_batter_preseason_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        assert len(transforms) > 0
        for tf in transforms:
            assert tf.lag == 1, f"{tf.name} should have lag=1 but has lag={tf.lag}"

    def test_statcast_transform_names(self) -> None:
        fs = build_batter_preseason_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "batted_ball" in transform_names
        assert "plate_discipline" in transform_names
        assert "expected_stats" in transform_names
        assert "spray_angle" in transform_names

    def test_different_version_from_true_talent(self) -> None:
        tt = build_batter_feature_set([2023])
        ps = build_batter_preseason_set([2023])
        assert tt.version != ps.version


class TestBatterPreseasonTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_batter_preseason_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_avg" in names
        assert "target_obp" in names
        assert "target_slg" in names

    def test_name(self) -> None:
        fs = build_batter_preseason_training_set([2023])
        assert fs.name == "statcast_gbm_batting_preseason_train"

    def test_statcast_transforms_are_lagged(self) -> None:
        fs = build_batter_preseason_training_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        for tf in transforms:
            assert tf.lag == 1


class TestBatterPreseasonFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = batter_preseason_feature_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_no_target_columns(self) -> None:
        columns = batter_preseason_feature_columns()
        assert not any(c.startswith("target_") for c in columns)


class TestPitcherPreseasonSet:
    def test_returns_feature_set(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_name(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        assert fs.name == "statcast_gbm_pitching_preseason"

    def test_has_pitcher_spine_filter(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="pitcher")

    def test_statcast_transforms_are_lagged(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        assert len(transforms) > 0
        for tf in transforms:
            assert tf.lag == 1

    def test_statcast_transform_names(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "pitch_mix" in transform_names
        assert "spin_profile" in transform_names
        assert "plate_discipline" in transform_names
        assert "batted_ball_against" in transform_names
        assert "command" in transform_names


class TestPitcherPreseasonTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_pitcher_preseason_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_era" in names
        assert "target_fip" in names

    def test_name(self) -> None:
        fs = build_pitcher_preseason_training_set([2023])
        assert fs.name == "statcast_gbm_pitching_preseason_train"

    def test_statcast_transforms_are_lagged(self) -> None:
        fs = build_pitcher_preseason_training_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        for tf in transforms:
            assert tf.lag == 1


class TestPitcherPreseasonFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = pitcher_preseason_feature_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_no_target_columns(self) -> None:
        columns = pitcher_preseason_feature_columns()
        assert not any(c.startswith("target_") for c in columns)


class TestLiveBatterCuratedColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = live_batter_curated_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_exact_columns(self) -> None:
        columns = live_batter_curated_columns()
        expected = [
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
        ]
        assert columns == expected

    def test_no_overlap_with_pruned(self) -> None:
        columns = set(live_batter_curated_columns())
        pruned = {
            "ld_pct",
            "avg_1",
            "slg_1",
            "obp_1",
            "age",
            "h_1",
            "pa_1",
            "triples_1",
            "sb_1",
            "hr_1",
            "bb_1",
            "k_pct_1",
            "bb_pct_1",
            "doubles_1",
            "so_1",
        }
        assert columns & pruned == set()

    def test_subset_of_full_columns(self) -> None:
        curated = set(live_batter_curated_columns())
        full = set(batter_feature_columns())
        assert curated <= full


class TestLivePitcherCuratedColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = live_pitcher_curated_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_exact_columns(self) -> None:
        columns = live_pitcher_curated_columns()
        expected = [
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
        assert columns == expected

    def test_no_overlap_with_pruned(self) -> None:
        columns = set(live_pitcher_curated_columns())
        pruned = {
            "ff_pct",
            "first_pitch_strike_pct",
            "bb_1",
            "si_velo",
            "ch_h_break",
            "sl_pct",
            "ff_h_break",
            "so_1",
            "si_pct",
            "ch_v_break",
            "ch_velo",
            "cu_velo",
            "cu_h_break",
            "sl_spin",
            "hr_1",
            "age",
            "cu_v_break",
            "cu_spin",
            "ch_spin",
            "sl_v_break",
            "ip_1",
            "era_1",
            "fip_1",
        }
        assert columns & pruned == set()

    def test_subset_of_full_columns(self) -> None:
        curated = set(live_pitcher_curated_columns())
        full = set(pitcher_feature_columns())
        assert curated <= full


class TestPreseasonBatterCuratedColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = preseason_batter_curated_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_exact_columns(self) -> None:
        columns = preseason_batter_curated_columns()
        expected = [
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
        assert columns == expected

    def test_no_overlap_with_pruned(self) -> None:
        columns = set(preseason_batter_curated_columns())
        pruned = {
            "chase_rate",
            "h_1",
            "xwoba",
            "whiff_rate",
            "triples_1",
            "hr_1",
            "sweet_spot_pct",
            "sb_1",
            "called_strike_pct",
            "hard_hit_pct",
            "avg_launch_angle",
        }
        assert columns & pruned == set()

    def test_subset_of_full_columns(self) -> None:
        curated = set(preseason_batter_curated_columns())
        full = set(batter_preseason_feature_columns())
        assert curated <= full


class TestPreseasonPitcherCuratedColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = preseason_pitcher_curated_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_exact_columns(self) -> None:
        columns = preseason_pitcher_curated_columns()
        expected = [
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
        assert columns == expected

    def test_no_overlap_with_pruned(self) -> None:
        columns = set(preseason_pitcher_curated_columns())
        pruned = {
            "sl_velo",
            "avg_extension",
            "sl_h_break",
            "hr_1",
            "si_velo",
            "fb_pct_against",
            "sl_v_break",
            "swinging_strike_pct",
        }
        assert columns & pruned == set()

    def test_subset_of_full_columns(self) -> None:
        curated = set(preseason_pitcher_curated_columns())
        full = set(pitcher_preseason_feature_columns())
        assert curated <= full


# --- Curated feature-set builder tests ---


class TestLiveBatterCuratedFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_live_batter_feature_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_name(self) -> None:
        fs = build_live_batter_feature_set([2023])
        assert fs.name == "statcast_gbm_batting_live"

    def test_has_batter_spine_filter(self) -> None:
        fs = build_live_batter_feature_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="batter")

    def test_no_age_feature(self) -> None:
        fs = build_live_batter_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "age" not in names

    def test_excludes_all_lags(self) -> None:
        fs = build_live_batter_feature_set([2023])
        names = [f.name for f in fs.features]
        for pruned in (
            "pa_1",
            "hr_1",
            "h_1",
            "doubles_1",
            "triples_1",
            "bb_1",
            "sb_1",
            "so_1",
        ):
            assert pruned not in names

    def test_includes_transforms(self) -> None:
        fs = build_live_batter_feature_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "batted_ball" in transform_names
        assert "plate_discipline" in transform_names
        assert "expected_stats" in transform_names
        assert "spray_angle" in transform_names


class TestLiveBatterCuratedTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_live_batter_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_avg" in names
        assert "target_hr" in names

    def test_name(self) -> None:
        fs = build_live_batter_training_set([2023])
        assert fs.name == "statcast_gbm_batting_live_train"

    def test_no_age_feature(self) -> None:
        fs = build_live_batter_training_set([2023])
        names = [f.name for f in fs.features]
        assert "age" not in names


class TestLivePitcherCuratedFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_live_pitcher_feature_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_name(self) -> None:
        fs = build_live_pitcher_feature_set([2023])
        assert fs.name == "statcast_gbm_pitching_live"

    def test_has_pitcher_spine_filter(self) -> None:
        fs = build_live_pitcher_feature_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="pitcher")

    def test_no_age_feature(self) -> None:
        fs = build_live_pitcher_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "age" not in names

    def test_excludes_all_lags(self) -> None:
        fs = build_live_pitcher_feature_set([2023])
        names = [f.name for f in fs.features]
        for pruned in ("so_1", "bb_1", "hr_1", "ip_1", "era_1", "fip_1"):
            assert pruned not in names

    def test_includes_transforms(self) -> None:
        fs = build_live_pitcher_feature_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "pitch_mix" in transform_names
        assert "spin_profile" in transform_names
        assert "plate_discipline" in transform_names
        assert "batted_ball_against" in transform_names
        assert "command" in transform_names


class TestLivePitcherCuratedTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_live_pitcher_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_era" in names
        assert "target_fip" in names

    def test_name(self) -> None:
        fs = build_live_pitcher_training_set([2023])
        assert fs.name == "statcast_gbm_pitching_live_train"


class TestPreseasonBatterCuratedFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_preseason_batter_curated_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_name(self) -> None:
        fs = build_preseason_batter_curated_set([2023])
        assert fs.name == "statcast_gbm_batting_preseason_curated"

    def test_includes_age(self) -> None:
        fs = build_preseason_batter_curated_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_excludes_pruned_lags(self) -> None:
        fs = build_preseason_batter_curated_set([2023])
        names = [f.name for f in fs.features]
        for pruned in ("h_1", "triples_1", "hr_1", "sb_1"):
            assert pruned not in names

    def test_includes_kept_lags(self) -> None:
        fs = build_preseason_batter_curated_set([2023])
        names = [f.name for f in fs.features]
        assert "pa_1" in names
        assert "so_1" in names

    def test_transforms_are_lagged(self) -> None:
        fs = build_preseason_batter_curated_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        assert len(transforms) > 0
        for tf in transforms:
            assert tf.lag == 1


class TestPreseasonBatterCuratedTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_preseason_batter_curated_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_avg" in names

    def test_name(self) -> None:
        fs = build_preseason_batter_curated_training_set([2023])
        assert fs.name == "statcast_gbm_batting_preseason_curated_train"


class TestPreseasonPitcherCuratedFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_preseason_pitcher_curated_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_name(self) -> None:
        fs = build_preseason_pitcher_curated_set([2023])
        assert fs.name == "statcast_gbm_pitching_preseason_curated"

    def test_includes_age(self) -> None:
        fs = build_preseason_pitcher_curated_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_excludes_hr_lag(self) -> None:
        fs = build_preseason_pitcher_curated_set([2023])
        names = [f.name for f in fs.features]
        assert "hr_1" not in names

    def test_includes_kept_lags(self) -> None:
        fs = build_preseason_pitcher_curated_set([2023])
        names = [f.name for f in fs.features]
        assert "ip_1" in names
        assert "so_1" in names
        assert "bb_1" in names

    def test_transforms_are_lagged(self) -> None:
        fs = build_preseason_pitcher_curated_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        assert len(transforms) > 0
        for tf in transforms:
            assert tf.lag == 1


class TestPreseasonPitcherCuratedTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_preseason_pitcher_curated_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_era" in names
        assert "target_fip" in names

    def test_name(self) -> None:
        fs = build_preseason_pitcher_curated_training_set([2023])
        assert fs.name == "statcast_gbm_pitching_preseason_curated_train"
