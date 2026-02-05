"""Tests for CSV projection source."""

from pathlib import Path

import pytest

from fantasy_baseball_manager.projections.csv_source import CSVProjectionSource
from fantasy_baseball_manager.projections.models import ProjectionSystem

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def batting_csv() -> Path:
    """Path to batting projections CSV fixture."""
    return FIXTURES_DIR / "steamer_2023_batting.csv"


@pytest.fixture
def pitching_csv() -> Path:
    """Path to pitching projections CSV fixture."""
    return FIXTURES_DIR / "steamer_2023_pitching.csv"


@pytest.fixture
def source(batting_csv: Path, pitching_csv: Path) -> CSVProjectionSource:
    """Create a CSV projection source with test fixtures."""
    return CSVProjectionSource(
        batting_path=batting_csv,
        pitching_path=pitching_csv,
        system=ProjectionSystem.STEAMER,
    )


class TestCSVProjectionSource:
    """Tests for CSVProjectionSource."""

    def test_fetch_projections_returns_projection_data(self, source: CSVProjectionSource) -> None:
        """fetch_projections returns ProjectionData with batting and pitching."""
        result = source.fetch_projections()

        assert result.system == ProjectionSystem.STEAMER
        assert len(result.batting) == 3
        assert len(result.pitching) == 3
        assert result.fetched_at is not None

    def test_parses_batting_projections(self, source: CSVProjectionSource) -> None:
        """Correctly parses batting projection fields from CSV."""
        result = source.fetch_projections()

        judge = result.batting[0]
        assert judge.player_id == "15640"
        assert judge.mlbam_id == "592450"
        assert judge.name == "Aaron Judge"
        assert judge.team == "NYY"
        assert judge.pa == 635
        assert judge.ab == 510
        assert judge.h == 145
        assert judge.singles == 78
        assert judge.doubles == 24
        assert judge.triples == 1
        assert judge.hr == 43
        assert judge.r == 110
        assert judge.rbi == 104
        assert judge.sb == 9
        assert judge.cs == 2
        assert judge.bb == 112
        assert judge.so == 156
        assert judge.hbp == 6
        assert judge.sf == 4
        assert judge.sh == 2
        assert judge.obp == pytest.approx(0.417, abs=0.001)
        assert judge.slg == pytest.approx(0.587, abs=0.001)
        assert judge.war == pytest.approx(6.7, abs=0.1)

    def test_parses_pitching_projections(self, source: CSVProjectionSource) -> None:
        """Correctly parses pitching projection fields from CSV."""
        result = source.fetch_projections()

        skubal = result.pitching[0]
        assert skubal.player_id == "22267"
        assert skubal.mlbam_id == "669373"
        assert skubal.name == "Tarik Skubal"
        assert skubal.team == "DET"
        assert skubal.g == 28
        assert skubal.gs == 28
        assert skubal.ip == pytest.approx(199.8, abs=0.1)
        assert skubal.w == 14
        assert skubal.l == 4
        assert skubal.sv == 0
        assert skubal.hld == 0
        assert skubal.so == 243
        assert skubal.bb == 40
        assert skubal.hbp == 7
        assert skubal.h == 136
        assert skubal.er == 62
        assert skubal.hr == 13
        assert skubal.era == pytest.approx(2.80, abs=0.01)
        assert skubal.whip == pytest.approx(1.02, abs=0.01)
        assert skubal.fip == pytest.approx(2.65, abs=0.01)
        assert skubal.war == pytest.approx(5.4, abs=0.1)

    def test_parses_reliever_saves_holds(self, source: CSVProjectionSource) -> None:
        """Correctly parses saves and holds for relievers."""
        result = source.fetch_projections()

        suarez = result.pitching[2]
        assert suarez.name == "Robert Suarez"
        assert suarez.sv == 35
        assert suarez.hld == 5
        assert suarez.gs == 0

    def test_handles_case_insensitive_headers(self, tmp_path: Path) -> None:
        """Handles CSV headers with different casing."""
        batting_path = tmp_path / "batting.csv"
        pitching_path = tmp_path / "pitching.csv"

        # Write CSV with lowercase headers
        batting_path.write_text(
            "name,team,playerid,mlbamid,g,pa,ab,h,1b,2b,3b,hr,r,rbi,sb,cs,bb,so,hbp,sf,sh,obp,slg,ops,woba,war\n"
            "Test Player,TST,12345,67890,100,400,350,90,50,20,5,15,60,50,10,3,40,80,5,3,2,0.350,0.450,0.800,0.360,3.0\n"
        )
        pitching_path.write_text(
            "name,team,playerid,mlbamid,g,gs,ip,w,l,sv,hld,so,bb,hbp,h,er,hr,era,whip,fip,war\n"
            "Test Pitcher,TST,54321,98765,30,20,150.0,10,5,0,0,150,40,5,120,50,12,3.00,1.07,3.20,2.5\n"
        )

        source = CSVProjectionSource(
            batting_path=batting_path,
            pitching_path=pitching_path,
            system=ProjectionSystem.STEAMER,
        )
        result = source.fetch_projections()

        assert result.batting[0].name == "Test Player"
        assert result.pitching[0].name == "Test Pitcher"

    def test_handles_idfg_column_name(self, tmp_path: Path) -> None:
        """Handles idfg as alternative to playerid column."""
        batting_path = tmp_path / "batting.csv"
        pitching_path = tmp_path / "pitching.csv"

        # Write CSV with idfg instead of playerid
        batting_path.write_text(
            "Name,Team,idfg,xMLBAMID,G,PA,AB,H,1B,2B,3B,HR,R,RBI,SB,CS,BB,SO,HBP,SF,SH,OBP,SLG,OPS,wOBA,WAR\n"
            "Test Player,TST,12345,67890,100,400,350,90,50,20,5,15,60,50,10,3,40,80,5,3,2,0.350,0.450,0.800,0.360,3.0\n"
        )
        pitching_path.write_text(
            "Name,Team,idfg,xMLBAMID,G,GS,IP,W,L,SV,HLD,SO,BB,HBP,H,ER,HR,ERA,WHIP,FIP,WAR\n"
            "Test Pitcher,TST,54321,98765,30,20,150.0,10,5,0,0,150,40,5,120,50,12,3.00,1.07,3.20,2.5\n"
        )

        source = CSVProjectionSource(
            batting_path=batting_path,
            pitching_path=pitching_path,
            system=ProjectionSystem.STEAMER,
        )
        result = source.fetch_projections()

        assert result.batting[0].player_id == "12345"
        assert result.pitching[0].player_id == "54321"

    def test_handles_xmlbamid_column_name(self, tmp_path: Path) -> None:
        """Handles xMLBAMID as alternative to MLBAMID column."""
        batting_path = tmp_path / "batting.csv"
        pitching_path = tmp_path / "pitching.csv"

        batting_path.write_text(
            "Name,Team,playerid,xMLBAMID,G,PA,AB,H,1B,2B,3B,HR,R,RBI,SB,CS,BB,SO,HBP,SF,SH,OBP,SLG,OPS,wOBA,WAR\n"
            "Test Player,TST,12345,67890,100,400,350,90,50,20,5,15,60,50,10,3,40,80,5,3,2,0.350,0.450,0.800,0.360,3.0\n"
        )
        pitching_path.write_text(
            "Name,Team,playerid,xMLBAMID,G,GS,IP,W,L,SV,HLD,SO,BB,HBP,H,ER,HR,ERA,WHIP,FIP,WAR\n"
            "Test Pitcher,TST,54321,98765,30,20,150.0,10,5,0,0,150,40,5,120,50,12,3.00,1.07,3.20,2.5\n"
        )

        source = CSVProjectionSource(
            batting_path=batting_path,
            pitching_path=pitching_path,
            system=ProjectionSystem.STEAMER,
        )
        result = source.fetch_projections()

        assert result.batting[0].mlbam_id == "67890"
        assert result.pitching[0].mlbam_id == "98765"

    def test_computes_singles_from_hits(self, tmp_path: Path) -> None:
        """Computes singles as H - 2B - 3B - HR when 1B column is missing."""
        batting_path = tmp_path / "batting.csv"
        pitching_path = tmp_path / "pitching.csv"

        # Write CSV without 1B column
        batting_path.write_text(
            "Name,Team,playerid,MLBAMID,G,PA,AB,H,2B,3B,HR,R,RBI,SB,CS,BB,SO,HBP,SF,SH,OBP,SLG,OPS,wOBA,WAR\n"
            "Test Player,TST,12345,67890,100,400,350,100,20,5,15,60,50,10,3,40,80,5,3,2,0.350,0.450,0.800,0.360,3.0\n"
        )
        pitching_path.write_text(
            "Name,Team,playerid,MLBAMID,G,GS,IP,W,L,SV,HLD,SO,BB,HBP,H,ER,HR,ERA,WHIP,FIP,WAR\n"
            "Test Pitcher,TST,54321,98765,30,20,150.0,10,5,0,0,150,40,5,120,50,12,3.00,1.07,3.20,2.5\n"
        )

        source = CSVProjectionSource(
            batting_path=batting_path,
            pitching_path=pitching_path,
            system=ProjectionSystem.STEAMER,
        )
        result = source.fetch_projections()

        # singles = 100 - 20 - 5 - 15 = 60
        assert result.batting[0].singles == 60

    def test_handles_missing_optional_fields(self, tmp_path: Path) -> None:
        """Handles missing optional fields with defaults."""
        batting_path = tmp_path / "batting.csv"
        pitching_path = tmp_path / "pitching.csv"

        # Minimal CSV with only required fields
        batting_path.write_text(
            "Name,Team,playerid,PA,AB,H,2B,3B,HR,BB,SO\n" "Test Player,TST,12345,400,350,100,20,5,15,40,80\n"
        )
        pitching_path.write_text("Name,Team,playerid,IP,SO,BB,H,ER\n" "Test Pitcher,TST,54321,150.0,150,40,120,50\n")

        source = CSVProjectionSource(
            batting_path=batting_path,
            pitching_path=pitching_path,
            system=ProjectionSystem.STEAMER,
        )
        result = source.fetch_projections()

        batter = result.batting[0]
        assert batter.mlbam_id is None
        assert batter.g == 0
        assert batter.r == 0
        assert batter.rbi == 0
        assert batter.sb == 0
        assert batter.cs == 0
        assert batter.hbp == 0
        assert batter.sf == 0
        assert batter.sh == 0
        assert batter.obp == 0.0
        assert batter.slg == 0.0
        assert batter.ops == 0.0
        assert batter.woba == 0.0
        assert batter.war == 0.0

        pitcher = result.pitching[0]
        assert pitcher.mlbam_id is None
        assert pitcher.g == 0
        assert pitcher.gs == 0
        assert pitcher.w == 0
        assert pitcher.l == 0
        assert pitcher.sv == 0
        assert pitcher.hld == 0
        assert pitcher.hbp == 0
        assert pitcher.hr == 0
        assert pitcher.fip == 0.0
        assert pitcher.war == 0.0

    def test_computes_era_whip_from_stats(self, tmp_path: Path) -> None:
        """Computes ERA and WHIP when not provided."""
        batting_path = tmp_path / "batting.csv"
        pitching_path = tmp_path / "pitching.csv"

        batting_path.write_text(
            "Name,Team,playerid,PA,AB,H,2B,3B,HR,BB,SO\n" "Test Player,TST,12345,400,350,100,20,5,15,40,80\n"
        )
        # No ERA or WHIP columns
        pitching_path.write_text("Name,Team,playerid,IP,SO,BB,H,ER\n" "Test Pitcher,TST,54321,90.0,100,30,70,30\n")

        source = CSVProjectionSource(
            batting_path=batting_path,
            pitching_path=pitching_path,
            system=ProjectionSystem.STEAMER,
        )
        result = source.fetch_projections()

        pitcher = result.pitching[0]
        # ERA = (30 / 90) * 9 = 3.00
        assert pitcher.era == pytest.approx(3.00, abs=0.01)
        # WHIP = (70 + 30) / 90 = 1.11
        assert pitcher.whip == pytest.approx(1.11, abs=0.01)

    def test_zips_system(self, batting_csv: Path, pitching_csv: Path) -> None:
        """Can parse ZiPS projections."""
        source = CSVProjectionSource(
            batting_path=batting_csv,
            pitching_path=pitching_csv,
            system=ProjectionSystem.ZIPS,
        )
        result = source.fetch_projections()

        assert result.system == ProjectionSystem.ZIPS

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing CSV file."""
        source = CSVProjectionSource(
            batting_path=tmp_path / "nonexistent.csv",
            pitching_path=tmp_path / "pitching.csv",
            system=ProjectionSystem.STEAMER,
        )

        with pytest.raises(FileNotFoundError):
            source.fetch_projections()
