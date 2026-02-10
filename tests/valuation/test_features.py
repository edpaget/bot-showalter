import math

import numpy as np
import pytest

from fantasy_baseball_manager.adp.training_dataset import BatterTrainingRow, PitcherTrainingRow
from fantasy_baseball_manager.valuation.features import (
    BATTER_FEATURE_NAMES,
    PITCHER_FEATURE_NAMES,
    batter_training_rows_to_arrays,
    batting_projection_to_features,
    pitcher_training_rows_to_arrays,
    pitching_projection_to_features,
    position_to_ordinal,
)


def _batter_row(
    player_id: str = "b1",
    name: str = "Batter",
    team: str = "NYY",
    year: int = 2024,
    position: str = "SS",
    pa: int = 600,
    hr: int = 30,
    r: int = 90,
    rbi: int = 100,
    sb: int = 15,
    bb: int = 60,
    so: int = 120,
    obp: float = 0.350,
    slg: float = 0.500,
    woba: float = 0.370,
    war: float = 5.0,
    adp: float = 25.0,
) -> BatterTrainingRow:
    return BatterTrainingRow(
        player_id=player_id,
        name=name,
        team=team,
        year=year,
        position=position,
        pa=pa,
        hr=hr,
        r=r,
        rbi=rbi,
        sb=sb,
        bb=bb,
        so=so,
        obp=obp,
        slg=slg,
        woba=woba,
        war=war,
        adp=adp,
    )


def _pitcher_row(
    player_id: str = "p1",
    name: str = "Pitcher",
    team: str = "LAD",
    year: int = 2024,
    ip: float = 200.0,
    w: int = 15,
    sv: int = 0,
    hld: int = 0,
    gs: int = 30,
    so: int = 200,
    bb: int = 50,
    h: int = 160,
    er: int = 60,
    hr: int = 20,
    era: float = 2.70,
    whip: float = 1.05,
    fip: float = 2.80,
    war: float = 5.0,
    adp: float = 30.0,
) -> PitcherTrainingRow:
    return PitcherTrainingRow(
        player_id=player_id,
        name=name,
        team=team,
        year=year,
        ip=ip,
        w=w,
        sv=sv,
        hld=hld,
        gs=gs,
        so=so,
        bb=bb,
        h=h,
        er=er,
        hr=hr,
        era=era,
        whip=whip,
        fip=fip,
        war=war,
        adp=adp,
    )


class TestPositionToOrdinal:
    def test_catcher_is_most_scarce(self) -> None:
        assert position_to_ordinal("C") == 1

    def test_shortstop(self) -> None:
        assert position_to_ordinal("SS") == 2

    def test_second_base(self) -> None:
        assert position_to_ordinal("2B") == 3

    def test_third_base(self) -> None:
        assert position_to_ordinal("3B") == 4

    def test_outfield_variants(self) -> None:
        assert position_to_ordinal("OF") == 5
        assert position_to_ordinal("CF") == 5
        assert position_to_ordinal("LF") == 5
        assert position_to_ordinal("RF") == 5

    def test_first_base_and_dh(self) -> None:
        assert position_to_ordinal("1B") == 6
        assert position_to_ordinal("DH") == 6

    def test_case_insensitive(self) -> None:
        assert position_to_ordinal("ss") == 2
        assert position_to_ordinal("Ss") == 2

    def test_unknown_position_defaults(self) -> None:
        assert position_to_ordinal("UTIL") == 5
        assert position_to_ordinal("") == 5


class TestBatterTrainingRowsToArrays:
    def test_shape(self) -> None:
        rows = [_batter_row(), _batter_row(player_id="b2")]
        X, y = batter_training_rows_to_arrays(rows)
        assert X.shape == (2, len(BATTER_FEATURE_NAMES))
        assert y.shape == (2,)

    def test_target_is_log_adp(self) -> None:
        rows = [_batter_row(adp=50.0)]
        _, y = batter_training_rows_to_arrays(rows)
        assert y[0] == pytest.approx(math.log(50.0))

    def test_features_match_expected(self) -> None:
        row = _batter_row(pa=500, hr=25, r=80, rbi=90, sb=10, bb=55, so=110, obp=0.340, slg=0.480, war=4.0, position="C")
        X, _ = batter_training_rows_to_arrays([row])
        assert X[0, 0] == 500  # pa
        assert X[0, 1] == 25  # hr
        assert X[0, 2] == 80  # r
        assert X[0, 3] == 90  # rbi
        assert X[0, 4] == 10  # sb
        assert X[0, 5] == 55  # bb
        assert X[0, 6] == 110  # so
        assert X[0, 7] == pytest.approx(0.340)  # obp
        assert X[0, 8] == pytest.approx(0.480)  # slg
        assert X[0, 9] == pytest.approx(4.0)  # war
        assert X[0, 10] == 1  # position = C

    def test_log_adp_roundtrips(self) -> None:
        adp = 42.0
        rows = [_batter_row(adp=adp)]
        _, y = batter_training_rows_to_arrays(rows)
        assert math.exp(y[0]) == pytest.approx(adp)

    def test_empty_rows(self) -> None:
        X, y = batter_training_rows_to_arrays([])
        assert X.shape == (0, len(BATTER_FEATURE_NAMES))
        assert y.shape == (0,)


class TestPitcherTrainingRowsToArrays:
    def test_shape(self) -> None:
        rows = [_pitcher_row(), _pitcher_row(player_id="p2")]
        X, y = pitcher_training_rows_to_arrays(rows)
        assert X.shape == (2, len(PITCHER_FEATURE_NAMES))
        assert y.shape == (2,)

    def test_target_is_log_adp(self) -> None:
        rows = [_pitcher_row(adp=100.0)]
        _, y = pitcher_training_rows_to_arrays(rows)
        assert y[0] == pytest.approx(math.log(100.0))

    def test_features_match_expected(self) -> None:
        row = _pitcher_row(ip=180.0, w=12, sv=5, hld=3, gs=28, so=190, bb=45, hr=18, era=3.00, whip=1.10)
        X, _ = pitcher_training_rows_to_arrays([row])
        assert X[0, 0] == pytest.approx(180.0)  # ip
        assert X[0, 1] == 12  # w
        assert X[0, 2] == 8  # nsvh (sv=5 + hld=3)
        assert X[0, 3] == 28  # gs
        assert X[0, 4] == 190  # so
        assert X[0, 5] == 45  # bb
        assert X[0, 6] == 18  # hr
        assert X[0, 7] == pytest.approx(3.00)  # era
        assert X[0, 8] == pytest.approx(1.10)  # whip

    def test_empty_rows(self) -> None:
        X, y = pitcher_training_rows_to_arrays([])
        assert X.shape == (0, len(PITCHER_FEATURE_NAMES))
        assert y.shape == (0,)


class TestBattingProjectionToFeatures:
    def test_shape(self) -> None:
        from fantasy_baseball_manager.marcel.models import BattingProjection

        proj = BattingProjection(
            player_id="b1", name="Hitter", year=2026, age=28,
            pa=600.0, ab=540.0, h=150.0, singles=100.0, doubles=30.0,
            triples=5.0, hr=15.0, bb=50.0, so=100.0, hbp=5.0,
            sf=3.0, sh=2.0, sb=20.0, cs=5.0, r=80.0, rbi=70.0,
        )
        features = batting_projection_to_features(proj, "SS")
        assert features.shape == (len(BATTER_FEATURE_NAMES),)
        assert features.dtype == np.float64

    def test_obp_computed(self) -> None:
        from fantasy_baseball_manager.marcel.models import BattingProjection

        proj = BattingProjection(
            player_id="b1", name="Hitter", year=2026, age=28,
            pa=500.0, ab=450.0, h=150.0, singles=100.0, doubles=30.0,
            triples=5.0, hr=15.0, bb=40.0, so=80.0, hbp=10.0,
            sf=3.0, sh=2.0, sb=10.0, cs=3.0, r=80.0, rbi=70.0,
        )
        features = batting_projection_to_features(proj)
        # OBP = (150 + 40 + 10) / 500 = 0.4
        assert features[7] == pytest.approx(0.4)

    def test_zero_pa(self) -> None:
        from fantasy_baseball_manager.marcel.models import BattingProjection

        proj = BattingProjection(
            player_id="b1", name="Zero", year=2026, age=28,
            pa=0.0, ab=0.0, h=0.0, singles=0.0, doubles=0.0,
            triples=0.0, hr=0.0, bb=0.0, so=0.0, hbp=0.0,
            sf=0.0, sh=0.0, sb=0.0, cs=0.0, r=0.0, rbi=0.0,
        )
        features = batting_projection_to_features(proj)
        assert features[7] == 0.0  # OBP
        assert features[8] == 0.0  # SLG


class TestPitchingProjectionToFeatures:
    def test_shape(self) -> None:
        from fantasy_baseball_manager.marcel.models import PitchingProjection

        proj = PitchingProjection(
            player_id="p1", name="Ace", year=2026, age=27,
            ip=200.0, g=33.0, gs=33.0, er=60.0, h=170.0,
            bb=50.0, so=200.0, hr=20.0, hbp=8.0,
            era=2.70, whip=1.10, w=15.0, nsvh=0.0,
        )
        features = pitching_projection_to_features(proj)
        assert features.shape == (len(PITCHER_FEATURE_NAMES),)
        assert features.dtype == np.float64

    def test_nsvh_and_gs_populated(self) -> None:
        from fantasy_baseball_manager.marcel.models import PitchingProjection

        proj = PitchingProjection(
            player_id="p1", name="Closer", year=2026, age=30,
            ip=65.0, g=60.0, gs=0.0, er=18.0, h=45.0,
            bb=20.0, so=80.0, hr=5.0, hbp=3.0,
            era=2.49, whip=1.00, w=4.0, nsvh=40.0,
        )
        features = pitching_projection_to_features(proj)
        assert features[2] == pytest.approx(40.0)  # nsvh
        assert features[3] == pytest.approx(0.0)  # gs

    def test_zero_ip(self) -> None:
        from fantasy_baseball_manager.marcel.models import PitchingProjection

        proj = PitchingProjection(
            player_id="p1", name="Zero", year=2026, age=27,
            ip=0.0, g=0.0, gs=0.0, er=0.0, h=0.0,
            bb=0.0, so=0.0, hr=0.0, hbp=0.0,
            era=0.0, whip=0.0, w=0.0, nsvh=0.0,
        )
        features = pitching_projection_to_features(proj)
        assert np.all(np.isfinite(features))
