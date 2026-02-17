from typing import Any

import pytest

from fantasy_baseball_manager.cli._output import (
    print_error,
    print_features,
    print_player_projections,
    print_player_valuations,
    print_valuation_rankings,
)
from fantasy_baseball_manager.domain.projection import PlayerProjection
from fantasy_baseball_manager.domain.valuation import PlayerValuation
from fantasy_baseball_manager.features.types import (
    DeltaFeature,
    Feature,
    Source,
    TransformFeature,
)


def _plain_feature() -> Feature:
    return Feature(name="hr_1", source=Source.BATTING, column="hr", lag=0)


def _lag_feature() -> Feature:
    return Feature(name="hr_prev", source=Source.BATTING, column="hr", lag=1)


def _system_feature() -> Feature:
    return Feature(name="proj_hr", source=Source.PROJECTION, column="hr", system="steamer")


def _computed_feature() -> Feature:
    return Feature(name="age", source=Source.PLAYER, column="", computed="age")


def _delta_feature() -> DeltaFeature:
    left = Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1)
    right = Feature(name="hr_2", source=Source.BATTING, column="hr", lag=2)
    return DeltaFeature(name="hr_diff", left=left, right=right)


def _transform_feature() -> TransformFeature:
    def dummy(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {}

    return TransformFeature(
        name="rate_calc",
        source=Source.STATCAST,
        columns=("ev", "la"),
        group_by=("player_id", "season"),
        transform=dummy,
        outputs=("barrel_pct", "hard_hit_pct"),
    )


class TestPrintError:
    def test_print_error_writes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_error("something went wrong")
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "something went wrong" in captured.err
        assert captured.out == ""


class TestPrintFeatures:
    def test_print_features_plain(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_plain_feature(),))
        captured = capsys.readouterr()
        assert "batting.hr" in captured.out
        assert "1 features" in captured.out

    def test_print_features_lag(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_lag_feature(),))
        captured = capsys.readouterr()
        assert "lag=1" in captured.out

    def test_print_features_system(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_system_feature(),))
        captured = capsys.readouterr()
        assert "system=steamer" in captured.out

    def test_print_features_computed(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_computed_feature(),))
        captured = capsys.readouterr()
        assert "computed=age" in captured.out

    def test_print_features_delta(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_delta_feature(),))
        captured = capsys.readouterr()
        assert "delta(hr_1 - hr_2)" in captured.out

    def test_print_features_transform(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_transform_feature(),))
        captured = capsys.readouterr()
        assert "transform" in captured.out
        assert "barrel_pct" in captured.out
        assert "hard_hit_pct" in captured.out

    def test_print_features_mixed(self, capsys: pytest.CaptureFixture[str]) -> None:
        features = (
            _plain_feature(),
            _lag_feature(),
            _system_feature(),
            _computed_feature(),
            _delta_feature(),
            _transform_feature(),
        )
        print_features("test_model", features)
        captured = capsys.readouterr()
        assert "6 features" in captured.out
        assert "batting.hr" in captured.out
        assert "lag=1" in captured.out
        assert "system=steamer" in captured.out
        assert "computed=age" in captured.out
        assert "delta(hr_1 - hr_2)" in captured.out
        assert "barrel_pct, hard_hit_pct" in captured.out


def _make_player_projection(
    stats: dict[str, Any],
    system: str = "steamer",
    version: str = "2025.1",
    source_type: str = "third_party",
    player_type: str = "batter",
) -> PlayerProjection:
    return PlayerProjection(
        player_name="Mike Trout",
        system=system,
        version=version,
        source_type=source_type,
        player_type=player_type,
        stats=stats,
    )


class TestPrintPlayerProjectionsLineage:
    def test_print_player_projections_ensemble_lineage(self, capsys: pytest.CaptureFixture[str]) -> None:
        proj = _make_player_projection(
            stats={
                "hr": 30.0,
                "rbi": 90.0,
                "_components": {"marcel": 0.6, "steamer": 0.4},
                "_mode": "weighted_average",
            },
            system="ensemble",
        )
        print_player_projections([proj])
        captured = capsys.readouterr()
        assert "Sources:" in captured.out
        assert "marcel 60%" in captured.out
        assert "steamer 40%" in captured.out
        assert "weighted_average" in captured.out

    def test_print_player_projections_composite_lineage(self, capsys: pytest.CaptureFixture[str]) -> None:
        proj = _make_player_projection(
            stats={
                "hr": 30.0,
                "_pt_system": "playing_time",
            },
            system="composite",
        )
        print_player_projections([proj])
        captured = capsys.readouterr()
        assert "PT source:" in captured.out
        assert "playing_time" in captured.out

    def test_print_player_projections_hides_metadata_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        proj = _make_player_projection(
            stats={
                "hr": 30.0,
                "avg": 0.280,
                "_components": {"marcel": 0.6, "steamer": 0.4},
                "_mode": "weighted_average",
                "_pt_system": "playing_time",
                "rates": {"hr_rate": 0.05},
            },
            system="ensemble",
        )
        print_player_projections([proj])
        captured = capsys.readouterr()
        assert "_components" not in captured.out
        assert "_mode" not in captured.out
        assert "_pt_system" not in captured.out
        assert "rates" not in captured.out
        assert "hr" in captured.out
        assert "avg" in captured.out

    def test_print_player_projections_plain_system(self, capsys: pytest.CaptureFixture[str]) -> None:
        proj = _make_player_projection(
            stats={"hr": 30.0, "avg": 0.280},
            system="steamer",
        )
        print_player_projections([proj])
        captured = capsys.readouterr()
        assert "Sources:" not in captured.out
        assert "PT source:" not in captured.out


def _make_player_valuation(
    player_name: str = "Juan Soto",
    system: str = "zar",
    version: str = "1.0",
    projection_system: str = "steamer",
    projection_version: str = "2025.1",
    player_type: str = "batter",
    position: str = "OF",
    value: float = 42.5,
    rank: int = 1,
    category_scores: dict[str, float] | None = None,
) -> PlayerValuation:
    return PlayerValuation(
        player_name=player_name,
        system=system,
        version=version,
        projection_system=projection_system,
        projection_version=projection_version,
        player_type=player_type,
        position=position,
        value=value,
        rank=rank,
        category_scores=category_scores or {"hr": 2.1, "sb": 0.5},
    )


class TestPrintPlayerValuations:
    def test_empty_valuations(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_player_valuations([])
        captured = capsys.readouterr()
        assert "No valuations found" in captured.out

    def test_single_valuation_shows_breakdown(self, capsys: pytest.CaptureFixture[str]) -> None:
        val = _make_player_valuation()
        print_player_valuations([val])
        captured = capsys.readouterr()
        assert "Juan Soto" in captured.out
        assert "zar" in captured.out
        assert "42.5" in captured.out
        assert "hr" in captured.out
        assert "2.1" in captured.out


class TestPrintValuationRankings:
    def test_empty_rankings(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_valuation_rankings([])
        captured = capsys.readouterr()
        assert "No valuations found" in captured.out

    def test_rankings_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        vals = [
            _make_player_valuation(player_name="Juan Soto", rank=1, value=42.5),
            _make_player_valuation(player_name="Aaron Judge", rank=2, value=38.0),
        ]
        print_valuation_rankings(vals)
        captured = capsys.readouterr()
        assert "Juan Soto" in captured.out
        assert "Aaron Judge" in captured.out
        assert "42.5" in captured.out
        assert "1" in captured.out
