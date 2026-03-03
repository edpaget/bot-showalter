from fantasy_baseball_manager.domain.temporal_stability import (
    StabilityResult,
    TargetStability,
)


class TestTargetStability:
    def test_construction(self) -> None:
        ts = TargetStability(
            target="slg",
            per_season_r=((2022, 0.70), (2023, 0.72)),
            mean_r=0.71,
            std_r=0.014,
            cv=0.02,
            classification="stable",
        )
        assert ts.target == "slg"
        assert ts.mean_r == 0.71
        assert ts.classification == "stable"

    def test_frozen(self) -> None:
        ts = TargetStability(
            target="slg",
            per_season_r=((2023, 0.70),),
            mean_r=0.70,
            std_r=0.0,
            cv=0.0,
            classification="stable",
        )
        try:
            ts.target = "avg"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised


class TestStabilityResult:
    def test_construction(self) -> None:
        ts = TargetStability(
            target="slg",
            per_season_r=((2023, 0.70),),
            mean_r=0.70,
            std_r=0.0,
            cv=0.0,
            classification="stable",
        )
        result = StabilityResult(
            column_spec="launch_speed",
            player_type="batter",
            seasons=(2023,),
            target_stabilities=(ts,),
        )
        assert result.column_spec == "launch_speed"
        assert result.player_type == "batter"
        assert result.seasons == (2023,)
        assert len(result.target_stabilities) == 1

    def test_frozen(self) -> None:
        result = StabilityResult(
            column_spec="launch_speed",
            player_type="batter",
            seasons=(2023,),
            target_stabilities=(),
        )
        try:
            result.column_spec = "barrel"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised

    def test_multiple_targets(self) -> None:
        stabilities = tuple(
            TargetStability(
                target=t,
                per_season_r=((2023, 0.5),),
                mean_r=0.5,
                std_r=0.0,
                cv=0.0,
                classification="stable",
            )
            for t in ("avg", "obp", "slg", "woba", "iso", "babip")
        )
        result = StabilityResult(
            column_spec="barrel",
            player_type="batter",
            seasons=(2023,),
            target_stabilities=stabilities,
        )
        assert len(result.target_stabilities) == 6
