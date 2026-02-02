from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.ros.projector import ROSProjector


def _make_batting_projection(
    player_id: str = "b1",
    name: str = "Batter One",
    pa: float = 600.0,
    hr: float = 30.0,
) -> BattingProjection:
    return BattingProjection(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        pa=pa,
        ab=500.0,
        h=150.0,
        singles=90.0,
        doubles=30.0,
        triples=5.0,
        hr=hr,
        bb=60.0,
        so=120.0,
        hbp=5.0,
        sf=4.0,
        sh=1.0,
        sb=10.0,
        cs=3.0,
        r=80.0,
        rbi=90.0,
    )


def _make_batting_actuals(
    player_id: str = "b1",
    name: str = "Batter One",
    pa: int = 300,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        pa=pa,
        ab=260,
        h=80,
        singles=50,
        doubles=15,
        triples=2,
        hr=13,
        bb=30,
        so=60,
        hbp=3,
        sf=2,
        sh=0,
        sb=5,
        cs=2,
        r=40,
        rbi=45,
    )


def _make_pitching_projection(
    player_id: str = "p1",
    name: str = "Pitcher One",
    ip: float = 180.0,
) -> PitchingProjection:
    return PitchingProjection(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        ip=ip,
        g=30.0,
        gs=30.0,
        er=60.0,
        h=150.0,
        bb=50.0,
        so=180.0,
        hr=20.0,
        hbp=8.0,
        era=3.00,
        whip=1.11,
        w=12.0,
        nsvh=0.0,
    )


def _make_pitching_actuals(
    player_id: str = "p1",
    name: str = "Pitcher One",
    ip: float = 90.0,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        ip=ip,
        g=15,
        gs=15,
        er=30,
        h=75,
        bb=25,
        so=90,
        hr=10,
        hbp=4,
        w=6,
        sv=0,
        hld=0,
        bs=0,
    )


class FakePipeline:
    """Fake pipeline that returns pre-configured projections."""

    def __init__(
        self,
        batters: list[BattingProjection] | None = None,
        pitchers: list[PitchingProjection] | None = None,
    ) -> None:
        self._batters = batters or []
        self._pitchers = pitchers or []

    def project_batters(self, data_source: object, year: int) -> list[BattingProjection]:
        return list(self._batters)

    def project_pitchers(self, data_source: object, year: int) -> list[PitchingProjection]:
        return list(self._pitchers)


class FakeDataSource:
    """Fake data source that returns pre-configured actuals."""

    def __init__(
        self,
        batting: list[BattingSeasonStats] | None = None,
        pitching: list[PitchingSeasonStats] | None = None,
    ) -> None:
        self._batting = batting or []
        self._pitching = pitching or []

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return list(self._batting)

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return list(self._pitching)

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return []

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return []


class FakeBlender:
    """Fake blender that marks projections as blended by setting hr to 999."""

    def __init__(self) -> None:
        self.batting_calls: list[tuple[BattingProjection, BattingSeasonStats]] = []
        self.pitching_calls: list[tuple[PitchingProjection, PitchingSeasonStats]] = []

    def blend_batting(
        self, preseason: BattingProjection, actuals: BattingSeasonStats,
    ) -> BattingProjection:
        self.batting_calls.append((preseason, actuals))
        return BattingProjection(
            player_id=preseason.player_id,
            name=preseason.name,
            year=preseason.year,
            age=preseason.age,
            pa=preseason.pa - actuals.pa,
            ab=preseason.ab,
            h=preseason.h,
            singles=preseason.singles,
            doubles=preseason.doubles,
            triples=preseason.triples,
            hr=999.0,  # sentinel
            bb=preseason.bb,
            so=preseason.so,
            hbp=preseason.hbp,
            sf=preseason.sf,
            sh=preseason.sh,
            sb=preseason.sb,
            cs=preseason.cs,
            r=preseason.r,
            rbi=preseason.rbi,
        )

    def blend_pitching(
        self, preseason: PitchingProjection, actuals: PitchingSeasonStats,
    ) -> PitchingProjection:
        self.pitching_calls.append((preseason, actuals))
        return PitchingProjection(
            player_id=preseason.player_id,
            name=preseason.name,
            year=preseason.year,
            age=preseason.age,
            ip=preseason.ip - actuals.ip,
            g=preseason.g,
            gs=preseason.gs,
            er=preseason.er,
            h=preseason.h,
            bb=preseason.bb,
            so=999.0,  # sentinel
            hr=preseason.hr,
            hbp=preseason.hbp,
            era=preseason.era,
            whip=preseason.whip,
            w=preseason.w,
            nsvh=preseason.nsvh,
        )


class TestROSProjectorBatting:
    def test_player_with_preseason_and_actuals_gets_blended(self) -> None:
        pipeline = FakePipeline(batters=[_make_batting_projection()])
        ds = FakeDataSource(batting=[_make_batting_actuals()])
        blender = FakeBlender()
        projector = ROSProjector(pipeline=pipeline, data_source=ds, blender=blender)

        results = projector.project_batters(2025)

        assert len(results) == 1
        assert results[0].hr == 999.0  # sentinel from fake blender

    def test_player_with_preseason_only_passes_through(self) -> None:
        preseason = _make_batting_projection(player_id="b1")
        pipeline = FakePipeline(batters=[preseason])
        ds = FakeDataSource(batting=[])  # no actuals
        blender = FakeBlender()
        projector = ROSProjector(pipeline=pipeline, data_source=ds, blender=blender)

        results = projector.project_batters(2025)

        assert len(results) == 1
        assert results[0] is preseason  # unchanged
        assert len(blender.batting_calls) == 0

    def test_player_with_actuals_only_excluded(self) -> None:
        pipeline = FakePipeline(batters=[])  # no preseason
        ds = FakeDataSource(batting=[_make_batting_actuals(player_id="b2")])
        blender = FakeBlender()
        projector = ROSProjector(pipeline=pipeline, data_source=ds, blender=blender)

        results = projector.project_batters(2025)

        assert len(results) == 0

    def test_multiple_players_blended_independently(self) -> None:
        pipeline = FakePipeline(batters=[
            _make_batting_projection(player_id="b1", name="A"),
            _make_batting_projection(player_id="b2", name="B"),
        ])
        ds = FakeDataSource(batting=[
            _make_batting_actuals(player_id="b1", name="A"),
            _make_batting_actuals(player_id="b2", name="B"),
        ])
        blender = FakeBlender()
        projector = ROSProjector(pipeline=pipeline, data_source=ds, blender=blender)

        results = projector.project_batters(2025)

        assert len(results) == 2
        assert len(blender.batting_calls) == 2

    def test_projector_calls_blender_not_hardcoded(self) -> None:
        pipeline = FakePipeline(batters=[_make_batting_projection()])
        ds = FakeDataSource(batting=[_make_batting_actuals()])
        blender = FakeBlender()
        projector = ROSProjector(pipeline=pipeline, data_source=ds, blender=blender)

        projector.project_batters(2025)

        assert len(blender.batting_calls) == 1
        pre, act = blender.batting_calls[0]
        assert pre.player_id == "b1"
        assert act.player_id == "b1"


class TestROSProjectorPitching:
    def test_player_with_preseason_and_actuals_gets_blended(self) -> None:
        pipeline = FakePipeline(pitchers=[_make_pitching_projection()])
        ds = FakeDataSource(pitching=[_make_pitching_actuals()])
        blender = FakeBlender()
        projector = ROSProjector(pipeline=pipeline, data_source=ds, blender=blender)

        results = projector.project_pitchers(2025)

        assert len(results) == 1
        assert results[0].so == 999.0

    def test_player_with_preseason_only_passes_through(self) -> None:
        preseason = _make_pitching_projection(player_id="p1")
        pipeline = FakePipeline(pitchers=[preseason])
        ds = FakeDataSource(pitching=[])
        blender = FakeBlender()
        projector = ROSProjector(pipeline=pipeline, data_source=ds, blender=blender)

        results = projector.project_pitchers(2025)

        assert len(results) == 1
        assert results[0] is preseason
        assert len(blender.pitching_calls) == 0

    def test_player_with_actuals_only_excluded(self) -> None:
        pipeline = FakePipeline(pitchers=[])
        ds = FakeDataSource(pitching=[_make_pitching_actuals(player_id="p2")])
        blender = FakeBlender()
        projector = ROSProjector(pipeline=pipeline, data_source=ds, blender=blender)

        results = projector.project_pitchers(2025)

        assert len(results) == 0

    def test_mixed_players(self) -> None:
        """One player matched, one unmatched — both appear in output."""
        pipeline = FakePipeline(pitchers=[
            _make_pitching_projection(player_id="p1"),
            _make_pitching_projection(player_id="p2"),
        ])
        ds = FakeDataSource(pitching=[_make_pitching_actuals(player_id="p1")])
        blender = FakeBlender()
        projector = ROSProjector(pipeline=pipeline, data_source=ds, blender=blender)

        results = projector.project_pitchers(2025)

        assert len(results) == 2
        assert len(blender.pitching_calls) == 1
        blended = [r for r in results if r.so == 999.0]
        passthrough = [r for r in results if r.so != 999.0]
        assert len(blended) == 1
        assert len(passthrough) == 1
