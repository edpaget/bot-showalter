from pathlib import Path

import pytest

from fantasy_baseball_manager.evaluation.csv_source import CsvProjectionSource


def _write_csv(path: Path, content: str) -> None:
    path.write_text(content)


class TestCsvProjectionSourceBatting:
    def test_basic_read(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "batting.csv"
        _write_csv(
            csv_path,
            "IDfg,Name,Age,PA,AB,H,1B,2B,3B,HR,BB,SO,HBP,SF,SH,SB,CS,R,RBI\n"
            "123,Mike Trout,31,600,540,160,100,30,5,25,50,120,5,3,2,10,3,80,90\n",
        )
        source = CsvProjectionSource(batting_path=csv_path, pitching_path=None)
        batters = source.batting_projections()
        assert len(batters) == 1
        b = batters[0]
        assert b.player_id == "123"
        assert b.name == "Mike Trout"
        assert b.hr == 25.0
        assert b.pa == 600.0
        assert b.singles == 100.0

    def test_singles_derived_when_absent(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "batting.csv"
        _write_csv(
            csv_path,
            "IDfg,Name,Age,PA,AB,H,2B,3B,HR,BB,SO,HBP,SF,SH,SB,CS\n"
            "123,Test,28,600,540,160,30,5,25,50,120,5,3,2,10,3\n",
        )
        source = CsvProjectionSource(batting_path=csv_path, pitching_path=None)
        batters = source.batting_projections()
        assert batters[0].singles == pytest.approx(100.0)  # 160 - 30 - 5 - 25

    def test_case_insensitive_headers(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "batting.csv"
        _write_csv(
            csv_path,
            "idfg,name,age,pa,ab,h,2b,3b,hr,bb,so,hbp,sf,sh,sb,cs\n"
            "123,Test,28,600,540,160,30,5,25,50,120,5,3,2,10,3\n",
        )
        source = CsvProjectionSource(batting_path=csv_path, pitching_path=None)
        batters = source.batting_projections()
        assert len(batters) == 1
        assert batters[0].hr == 25.0

    def test_none_path_returns_empty(self) -> None:
        source = CsvProjectionSource(batting_path=None, pitching_path=None)
        assert source.batting_projections() == []

    def test_missing_file_raises(self) -> None:
        source = CsvProjectionSource(batting_path=Path("/nonexistent/batting.csv"), pitching_path=None)
        with pytest.raises(FileNotFoundError):
            source.batting_projections()


class TestCsvProjectionSourcePitching:
    def test_era_whip_from_components(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pitching.csv"
        _write_csv(
            csv_path,
            "IDfg,Name,Age,IP,G,GS,ER,H,BB,SO,HR,HBP\n" "456,Ace,27,180,32,32,60,150,50,200,20,5\n",
        )
        source = CsvProjectionSource(batting_path=None, pitching_path=csv_path)
        pitchers = source.pitching_projections()
        assert len(pitchers) == 1
        p = pitchers[0]
        assert p.player_id == "456"
        assert p.era == pytest.approx(60 / 180 * 9)
        assert p.whip == pytest.approx((150 + 50) / 180)
        assert p.so == 200.0
