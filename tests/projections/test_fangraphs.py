"""Tests for FanGraphs projection fetcher."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from fantasy_baseball_manager.projections.fangraphs import FanGraphsProjectionSource
from fantasy_baseball_manager.projections.models import ProjectionSystem

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def batting_json() -> list[dict]:
    """Load batting projections fixture."""
    with open(FIXTURES_DIR / "steamer_batting.json") as f:
        return json.load(f)


@pytest.fixture
def pitching_json() -> list[dict]:
    """Load pitching projections fixture."""
    with open(FIXTURES_DIR / "steamer_pitching.json") as f:
        return json.load(f)


class TestFanGraphsProjectionSource:
    """Tests for FanGraphsProjectionSource."""

    def test_fetch_projections_returns_projection_data(
        self, batting_json: list[dict], pitching_json: list[dict]
    ) -> None:
        """fetch_projections returns ProjectionData with batting and pitching."""
        source = FanGraphsProjectionSource(system=ProjectionSystem.STEAMER)

        with patch.object(source, "_fetch_json") as mock_fetch:
            mock_fetch.side_effect = [batting_json, pitching_json]

            result = source.fetch_projections()

        assert result.system == ProjectionSystem.STEAMER
        assert len(result.batting) == 3
        assert len(result.pitching) == 3
        assert result.fetched_at is not None

    def test_parses_batting_projections(
        self, batting_json: list[dict], pitching_json: list[dict]
    ) -> None:
        """Correctly parses batting projection fields."""
        source = FanGraphsProjectionSource(system=ProjectionSystem.STEAMER)

        with patch.object(source, "_fetch_json") as mock_fetch:
            mock_fetch.side_effect = [batting_json, pitching_json]

            result = source.fetch_projections()

        judge = result.batting[0]
        assert judge.player_id == "15640"
        assert judge.mlbam_id == "592450"
        assert judge.name == "Aaron Judge"
        assert judge.team == "NYY"
        assert judge.position == "OF"
        assert judge.pa == 635
        assert judge.singles == 78
        assert judge.doubles == 24
        assert judge.triples == 1
        assert judge.hr == 43
        assert judge.r == 110
        assert judge.rbi == 104
        assert judge.sb == 9
        assert judge.bb == 112
        assert judge.sf == 4
        assert judge.sh == 2
        assert judge.obp == pytest.approx(0.417, abs=0.001)
        assert judge.war == pytest.approx(6.7, abs=0.1)

    def test_parses_pitching_projections(
        self, batting_json: list[dict], pitching_json: list[dict]
    ) -> None:
        """Correctly parses pitching projection fields."""
        source = FanGraphsProjectionSource(system=ProjectionSystem.STEAMER)

        with patch.object(source, "_fetch_json") as mock_fetch:
            mock_fetch.side_effect = [batting_json, pitching_json]

            result = source.fetch_projections()

        skubal = result.pitching[0]
        assert skubal.player_id == "22267"
        assert skubal.mlbam_id == "669373"
        assert skubal.name == "Tarik Skubal"
        assert skubal.team == "DET"
        assert skubal.ip == pytest.approx(199.8, abs=0.1)
        assert skubal.w == 14
        assert skubal.sv == 0
        assert skubal.so == 243
        assert skubal.hbp == 7
        assert skubal.era == pytest.approx(2.80, abs=0.01)
        assert skubal.whip == pytest.approx(1.02, abs=0.01)

    def test_parses_reliever_saves_holds(
        self, batting_json: list[dict], pitching_json: list[dict]
    ) -> None:
        """Correctly parses saves and holds for relievers."""
        source = FanGraphsProjectionSource(system=ProjectionSystem.STEAMER)

        with patch.object(source, "_fetch_json") as mock_fetch:
            mock_fetch.side_effect = [batting_json, pitching_json]

            result = source.fetch_projections()

        suarez = result.pitching[2]
        assert suarez.name == "Robert Suarez"
        assert suarez.sv == 35
        assert suarez.hld == 5
        assert suarez.gs == 0

    def test_handles_missing_mlbam_id(
        self, batting_json: list[dict], pitching_json: list[dict]
    ) -> None:
        """Handles missing xMLBAMID gracefully."""
        batting_json[0]["xMLBAMID"] = None
        source = FanGraphsProjectionSource(system=ProjectionSystem.STEAMER)

        with patch.object(source, "_fetch_json") as mock_fetch:
            mock_fetch.side_effect = [batting_json, pitching_json]

            result = source.fetch_projections()

        assert result.batting[0].mlbam_id is None

    def test_zips_system(
        self, batting_json: list[dict], pitching_json: list[dict]
    ) -> None:
        """Can fetch ZiPS projections."""
        source = FanGraphsProjectionSource(system=ProjectionSystem.ZIPS)

        with patch.object(source, "_fetch_json") as mock_fetch:
            mock_fetch.side_effect = [batting_json, pitching_json]

            result = source.fetch_projections()

        assert result.system == ProjectionSystem.ZIPS

    def test_constructs_correct_urls(self) -> None:
        """Constructs correct API URLs for each system."""
        steamer = FanGraphsProjectionSource(system=ProjectionSystem.STEAMER)
        zips = FanGraphsProjectionSource(system=ProjectionSystem.ZIPS)

        assert "type=steamer" in steamer._batting_url()
        assert "type=steamer" in steamer._pitching_url()
        assert "type=zips" in zips._batting_url()
        assert "type=zips" in zips._pitching_url()
