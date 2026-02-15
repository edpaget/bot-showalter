import pytest

from fantasy_baseball_manager.domain.minor_league_batting_stats import MinorLeagueBattingStats


class TestMinorLeagueBattingStats:
    def test_construct_with_required_fields(self) -> None:
        stats = MinorLeagueBattingStats(
            player_id=1,
            season=2024,
            level="AAA",
            league="International League",
            team="Syracuse Mets",
            g=120,
            pa=500,
            ab=450,
            h=130,
            doubles=25,
            triples=3,
            hr=18,
            r=70,
            rbi=65,
            bb=40,
            so=100,
            sb=15,
            cs=5,
            avg=0.289,
            obp=0.350,
            slg=0.480,
            age=24.5,
        )
        assert stats.player_id == 1
        assert stats.season == 2024
        assert stats.level == "AAA"
        assert stats.league == "International League"
        assert stats.team == "Syracuse Mets"
        assert stats.g == 120
        assert stats.pa == 500
        assert stats.ab == 450
        assert stats.h == 130
        assert stats.doubles == 25
        assert stats.triples == 3
        assert stats.hr == 18
        assert stats.r == 70
        assert stats.rbi == 65
        assert stats.bb == 40
        assert stats.so == 100
        assert stats.sb == 15
        assert stats.cs == 5
        assert stats.avg == pytest.approx(0.289)
        assert stats.obp == pytest.approx(0.350)
        assert stats.slg == pytest.approx(0.480)
        assert stats.age == pytest.approx(24.5)

    def test_optional_fields_default_to_none(self) -> None:
        stats = MinorLeagueBattingStats(
            player_id=1,
            season=2024,
            level="AA",
            league="Eastern League",
            team="Binghamton Rumble Ponies",
            g=100,
            pa=400,
            ab=360,
            h=90,
            doubles=20,
            triples=2,
            hr=12,
            r=50,
            rbi=45,
            bb=30,
            so=80,
            sb=10,
            cs=3,
            avg=0.250,
            obp=0.320,
            slg=0.420,
            age=22.0,
        )
        assert stats.id is None
        assert stats.hbp is None
        assert stats.sf is None
        assert stats.sh is None
        assert stats.loaded_at is None

    def test_frozen(self) -> None:
        stats = MinorLeagueBattingStats(
            player_id=1,
            season=2024,
            level="A+",
            league="South Atlantic League",
            team="Brooklyn Cyclones",
            g=80,
            pa=300,
            ab=270,
            h=70,
            doubles=15,
            triples=1,
            hr=8,
            r=35,
            rbi=30,
            bb=25,
            so=60,
            sb=8,
            cs=2,
            avg=0.259,
            obp=0.330,
            slg=0.400,
            age=21.0,
        )
        with pytest.raises(AttributeError):
            stats.hr = 99  # type: ignore[misc]
