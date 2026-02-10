"""Tests for training dataset builder."""

from fantasy_baseball_manager.adp.models import ADPEntry
from fantasy_baseball_manager.adp.training_dataset import build_training_dataset
from fantasy_baseball_manager.projections.models import BattingProjection, PitchingProjection


def _batter(
    name: str = "Mike Trout",
    player_id: str = "123",
    pa: int = 500,
    **kwargs: object,
) -> BattingProjection:
    """Create a BattingProjection with reasonable defaults."""
    defaults: dict[str, object] = dict(
        player_id=player_id,
        mlbam_id=None,
        name=name,
        team="LAA",
        position="CF",
        g=140,
        pa=pa,
        ab=450,
        h=130,
        singles=80,
        doubles=25,
        triples=3,
        hr=22,
        r=80,
        rbi=70,
        sb=15,
        cs=3,
        bb=60,
        so=100,
        hbp=5,
        sf=4,
        sh=0,
        obp=0.350,
        slg=0.480,
        ops=0.830,
        woba=0.360,
        war=4.0,
    )
    defaults.update(kwargs)
    return BattingProjection(**defaults)  # type: ignore[arg-type]


def _pitcher(
    name: str = "Gerrit Cole",
    player_id: str = "456",
    ip: float = 180.0,
    **kwargs: object,
) -> PitchingProjection:
    """Create a PitchingProjection with reasonable defaults."""
    defaults: dict[str, object] = dict(
        player_id=player_id,
        mlbam_id=None,
        name=name,
        team="NYY",
        g=30,
        gs=30,
        ip=ip,
        w=14,
        l=6,
        sv=0,
        hld=0,
        so=220,
        bb=40,
        hbp=5,
        h=140,
        er=60,
        hr=18,
        era=3.00,
        whip=1.00,
        fip=3.10,
        war=5.0,
    )
    defaults.update(kwargs)
    return PitchingProjection(**defaults)  # type: ignore[arg-type]


def _adp(name: str, adp: float, positions: tuple[str, ...] = ("OF",)) -> ADPEntry:
    return ADPEntry(name=name, adp=adp, positions=positions)


class TestBuildTrainingDataset:
    """Tests for build_training_dataset."""

    def test_join_by_exact_normalized_name(self) -> None:
        batting = [_batter(name="Mike Trout", player_id="1")]
        adp_entries = [_adp("Mike Trout", 5.0)]
        result = build_training_dataset(batting, [], adp_entries, year=2024)

        assert len(result.batter_rows) == 1
        assert result.batter_rows[0].name == "Mike Trout"
        assert result.batter_rows[0].adp == 5.0
        assert result.batter_rows[0].year == 2024

    def test_accent_mismatch_still_joins(self) -> None:
        """Projection has plain ASCII, ADP has accented name."""
        batting = [_batter(name="Jose Ramirez", player_id="2")]
        adp_entries = [_adp("José Ramírez", 8.0)]
        result = build_training_dataset(batting, [], adp_entries, year=2024)

        assert len(result.batter_rows) == 1
        assert result.batter_rows[0].adp == 8.0

    def test_two_way_player_matches_both(self) -> None:
        batting = [_batter(name="Shohei Ohtani", player_id="10", pa=600)]
        pitching = [_pitcher(name="Shohei Ohtani", player_id="10", ip=150.0)]
        adp_entries = [_adp("Shohei Ohtani", 1.0, ("DH", "SP"))]
        result = build_training_dataset(batting, pitching, adp_entries, year=2024)

        assert len(result.batter_rows) == 1
        assert len(result.pitcher_rows) == 1
        assert result.batter_rows[0].adp == 1.0
        assert result.pitcher_rows[0].adp == 1.0

    def test_min_pa_filtering(self) -> None:
        batting = [
            _batter(name="Star Player", player_id="1", pa=500),
            _batter(name="Bench Player", player_id="2", pa=30),
        ]
        adp_entries = [_adp("Star Player", 5.0), _adp("Bench Player", 300.0)]
        result = build_training_dataset(batting, [], adp_entries, year=2024, min_pa=50)

        assert len(result.batter_rows) == 1
        assert result.batter_rows[0].name == "Star Player"

    def test_min_ip_filtering(self) -> None:
        pitching = [
            _pitcher(name="Ace Pitcher", player_id="1", ip=180.0),
            _pitcher(name="Mop-up Guy", player_id="2", ip=10.0),
        ]
        adp_entries = [_adp("Ace Pitcher", 15.0), _adp("Mop-up Guy", 400.0)]
        result = build_training_dataset([], pitching, adp_entries, year=2024, min_ip=20.0)

        assert len(result.pitcher_rows) == 1
        assert result.pitcher_rows[0].name == "Ace Pitcher"

    def test_unmatched_adp_tracked(self) -> None:
        batting = [_batter(name="Known Player", player_id="1")]
        adp_entries = [_adp("Known Player", 5.0), _adp("Mystery Player", 200.0)]
        result = build_training_dataset(batting, [], adp_entries, year=2024)

        assert "Mystery Player" in result.unmatched_adp

    def test_unmatched_batting_tracked(self) -> None:
        batting = [_batter(name="Projected Only", player_id="1")]
        adp_entries: list[ADPEntry] = []
        result = build_training_dataset(batting, [], adp_entries, year=2024)

        assert "Projected Only" in result.unmatched_batting

    def test_unmatched_pitching_tracked(self) -> None:
        pitching = [_pitcher(name="Projected Pitcher", player_id="1")]
        adp_entries: list[ADPEntry] = []
        result = build_training_dataset([], pitching, adp_entries, year=2024)

        assert "Projected Pitcher" in result.unmatched_pitching

    def test_empty_inputs_empty_result(self) -> None:
        result = build_training_dataset([], [], [], year=2024)

        assert result.batter_rows == []
        assert result.pitcher_rows == []
        assert result.unmatched_adp == []
        assert result.unmatched_batting == []
        assert result.unmatched_pitching == []

    def test_batter_row_contains_projection_stats(self) -> None:
        batting = [_batter(name="Mike Trout", player_id="1", hr=30, sb=20)]
        adp_entries = [_adp("Mike Trout", 5.0)]
        result = build_training_dataset(batting, [], adp_entries, year=2024)

        row = result.batter_rows[0]
        assert row.player_id == "1"
        assert row.hr == 30
        assert row.sb == 20

    def test_pitcher_row_contains_projection_stats(self) -> None:
        pitching = [_pitcher(name="Gerrit Cole", player_id="2", so=220, era=3.0)]
        adp_entries = [_adp("Gerrit Cole", 10.0)]
        result = build_training_dataset([], pitching, adp_entries, year=2024)

        row = result.pitcher_rows[0]
        assert row.player_id == "2"
        assert row.so == 220
        assert row.era == 3.0
