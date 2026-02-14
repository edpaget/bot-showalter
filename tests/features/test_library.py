from fantasy_baseball_manager.features import delta, projection
from fantasy_baseball_manager.features.library import (
    PLAYER_METADATA,
    STANDARD_BATTING_COUNTING,
    STANDARD_BATTING_RATES,
)
from fantasy_baseball_manager.features.types import DeltaFeature, Feature, Source


class TestStandardBattingCounting:
    def test_length(self) -> None:
        # 10 stats x 3 lags = 30
        assert len(STANDARD_BATTING_COUNTING) == 30

    def test_all_are_features(self) -> None:
        for f in STANDARD_BATTING_COUNTING:
            assert isinstance(f, Feature)

    def test_all_batting_source(self) -> None:
        for f in STANDARD_BATTING_COUNTING:
            assert f.source == Source.BATTING

    def test_lags_present(self) -> None:
        lags = {f.lag for f in STANDARD_BATTING_COUNTING}
        assert lags == {1, 2, 3}

    def test_stats_present(self) -> None:
        columns = {f.column for f in STANDARD_BATTING_COUNTING}
        assert columns == {"pa", "ab", "h", "hr", "rbi", "r", "sb", "cs", "bb", "so"}

    def test_names_follow_pattern(self) -> None:
        for f in STANDARD_BATTING_COUNTING:
            assert f.name == f"{f.column}_{f.lag}"


class TestStandardBattingRates:
    def test_length(self) -> None:
        assert len(STANDARD_BATTING_RATES) == 6

    def test_all_lag_1(self) -> None:
        for f in STANDARD_BATTING_RATES:
            assert f.lag == 1

    def test_rate_columns(self) -> None:
        columns = {f.column for f in STANDARD_BATTING_RATES}
        assert columns == {"avg", "obp", "slg", "ops", "woba", "wrc_plus"}


class TestPlayerMetadata:
    def test_length(self) -> None:
        assert len(PLAYER_METADATA) == 1

    def test_age_feature(self) -> None:
        assert PLAYER_METADATA[0].name == "age"
        assert PLAYER_METADATA[0].computed == "age"


class TestProjectionSourceRef:
    def test_projection_col(self) -> None:
        feature = projection.col("hr").system("steamer").alias("steamer_hr")
        assert feature.source == Source.PROJECTION
        assert feature.column == "hr"
        assert feature.system == "steamer"
        assert feature.name == "steamer_hr"


class TestDeltaHelper:
    def test_delta_creates_delta_feature(self) -> None:
        left = Feature(name="actual_hr", source=Source.BATTING, column="hr", lag=0)
        right = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer")
        d = delta("hr_error", left, right)
        assert isinstance(d, DeltaFeature)
        assert d.name == "hr_error"
        assert d.left is left
        assert d.right is right
