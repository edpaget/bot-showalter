from fantasy_baseball_manager.models.mle.types import TranslatedBattingLine


def _make_line(**overrides: int) -> TranslatedBattingLine:
    defaults: dict[str, int] = {
        "player_id": 1,
        "season": 2024,
        "pa": 500,
        "ab": 450,
        "h": 120,
        "doubles": 25,
        "triples": 5,
        "hr": 15,
        "bb": 40,
        "so": 100,
        "hbp": 5,
        "sf": 5,
    }
    defaults.update(overrides)
    return TranslatedBattingLine(source_level="AAA", **defaults)


class TestAvg:
    def test_normal(self) -> None:
        line = _make_line(h=120, ab=450)
        assert line.avg == 120 / 450

    def test_zero_ab(self) -> None:
        line = _make_line(ab=0, pa=0)
        assert line.avg == 0.0


class TestObp:
    def test_normal(self) -> None:
        line = _make_line(h=120, bb=40, hbp=5, ab=450, sf=5)
        assert line.obp == (120 + 40 + 5) / (450 + 40 + 5 + 5)

    def test_zero_denominator(self) -> None:
        line = _make_line(ab=0, bb=0, hbp=0, sf=0, pa=0, h=0)
        assert line.obp == 0.0


class TestSlg:
    def test_normal(self) -> None:
        line = _make_line(h=120, doubles=25, triples=5, hr=15, ab=450)
        tb = 120 + 25 + 5 * 2 + 15 * 3
        assert line.slg == tb / 450

    def test_zero_ab(self) -> None:
        line = _make_line(ab=0, pa=0)
        assert line.slg == 0.0


class TestIso:
    def test_normal(self) -> None:
        line = _make_line(h=120, doubles=25, triples=5, hr=15, ab=450)
        assert line.iso == line.slg - line.avg

    def test_zero_ab(self) -> None:
        line = _make_line(ab=0, pa=0)
        assert line.iso == 0.0


class TestKPct:
    def test_normal(self) -> None:
        line = _make_line(so=100, pa=500)
        assert line.k_pct == 100 / 500

    def test_zero_pa(self) -> None:
        line = _make_line(pa=0, ab=0)
        assert line.k_pct == 0.0


class TestBbPct:
    def test_normal(self) -> None:
        line = _make_line(bb=40, pa=500)
        assert line.bb_pct == 40 / 500

    def test_zero_pa(self) -> None:
        line = _make_line(pa=0, ab=0)
        assert line.bb_pct == 0.0


class TestBabip:
    def test_normal(self) -> None:
        line = _make_line(h=120, hr=15, ab=450, so=100, sf=5)
        bip = 450 - 100 - 15 + 5
        assert line.babip == (120 - 15) / bip

    def test_zero_bip(self) -> None:
        # bip = ab - so - hr + sf = 0 - 0 - 0 + 0 = 0
        line = _make_line(ab=0, so=0, hr=0, sf=0, pa=0, h=0)
        assert line.babip == 0.0
