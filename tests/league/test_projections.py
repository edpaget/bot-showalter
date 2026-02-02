import pytest

from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamRoster
from fantasy_baseball_manager.league.projections import match_projections
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection


class FakeIdMapper:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._yahoo_to_fg = mapping
        self._fg_to_yahoo = {v: k for k, v in mapping.items()}

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return self._yahoo_to_fg.get(yahoo_id)

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return self._fg_to_yahoo.get(fangraphs_id)


def _make_batter_projection(
    player_id: str = "fg1",
    name: str = "Batter",
    pa: float = 600.0,
    ab: float = 540.0,
    h: float = 160.0,
    hr: float = 30.0,
    bb: float = 50.0,
    hbp: float = 5.0,
    sb: float = 10.0,
) -> BattingProjection:
    return BattingProjection(
        player_id=player_id,
        name=name,
        year=2026,
        age=28,
        pa=pa,
        ab=ab,
        h=h,
        singles=h - hr - 30 - 5,
        doubles=30.0,
        triples=5.0,
        hr=hr,
        bb=bb,
        so=120.0,
        hbp=hbp,
        sf=3.0,
        sh=2.0,
        sb=sb,
        cs=3.0,
        r=0.0,
        rbi=0.0,
    )


def _make_pitcher_projection(
    player_id: str = "fgp1",
    name: str = "Pitcher",
    ip: float = 180.0,
    er: float = 60.0,
    h: float = 150.0,
    bb: float = 50.0,
    so: float = 200.0,
) -> PitchingProjection:
    return PitchingProjection(
        player_id=player_id,
        name=name,
        year=2026,
        age=28,
        ip=ip,
        g=32.0,
        gs=32.0,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=20.0,
        hbp=5.0,
        era=er / ip * 9 if ip > 0 else 0,
        whip=(h + bb) / ip if ip > 0 else 0,
        w=0.0,
        nsvh=0.0,
    )


def _roster_player(yahoo_id: str, name: str, position_type: str) -> RosterPlayer:
    return RosterPlayer(
        yahoo_id=yahoo_id,
        name=name,
        position_type=position_type,
        eligible_positions=("Util",),
    )


class TestMatchProjections:
    def test_matched_batter_totals(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Team A",
                    players=(
                        _roster_player("y1", "Batter 1", "B"),
                        _roster_player("y2", "Batter 2", "B"),
                    ),
                ),
            ),
        )
        batting = [
            _make_batter_projection(player_id="fg1", hr=25.0, sb=10.0, h=150.0, pa=550.0, ab=500.0, bb=40.0, hbp=5.0),
            _make_batter_projection(player_id="fg2", hr=35.0, sb=5.0, h=170.0, pa=650.0, ab=580.0, bb=60.0, hbp=5.0),
        ]
        mapper = FakeIdMapper({"y1": "fg1", "y2": "fg2"})

        result = match_projections(rosters, batting, [], mapper)

        assert len(result) == 1
        team = result[0]
        assert team.total_hr == pytest.approx(60.0)
        assert team.total_sb == pytest.approx(15.0)
        assert team.total_h == pytest.approx(320.0)
        assert team.total_pa == pytest.approx(1200.0)
        assert team.unmatched_count == 0

    def test_matched_pitcher_totals(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Team A",
                    players=(_roster_player("yp1", "Pitcher 1", "P"),),
                ),
            ),
        )
        pitching = [_make_pitcher_projection(player_id="fgp1", ip=180.0, er=60.0, h=150.0, bb=50.0, so=200.0)]
        mapper = FakeIdMapper({"yp1": "fgp1"})

        result = match_projections(rosters, [], pitching, mapper)

        team = result[0]
        assert team.total_ip == pytest.approx(180.0)
        assert team.total_so == pytest.approx(200.0)
        assert team.team_era == pytest.approx(3.0)
        assert team.team_whip == pytest.approx(200.0 / 180.0)

    def test_unmatched_player_counted(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Team A",
                    players=(
                        _roster_player("y1", "Known", "B"),
                        _roster_player("y_unknown", "Unknown", "B"),
                    ),
                ),
            ),
        )
        batting = [_make_batter_projection(player_id="fg1")]
        mapper = FakeIdMapper({"y1": "fg1"})

        result = match_projections(rosters, batting, [], mapper)

        assert result[0].unmatched_count == 1
        matched_players = [p for p in result[0].players if p.matched]
        assert len(matched_players) == 1

    def test_weighted_team_avg(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Team A",
                    players=(
                        _roster_player("y1", "B1", "B"),
                        _roster_player("y2", "B2", "B"),
                    ),
                ),
            ),
        )
        batting = [
            _make_batter_projection(player_id="fg1", h=100.0, ab=400.0, pa=450.0, bb=40.0, hbp=5.0),
            _make_batter_projection(player_id="fg2", h=200.0, ab=600.0, pa=650.0, bb=40.0, hbp=5.0),
        ]
        mapper = FakeIdMapper({"y1": "fg1", "y2": "fg2"})

        result = match_projections(rosters, batting, [], mapper)

        team = result[0]
        expected_avg = 300.0 / 1000.0
        expected_obp = (300.0 + 80.0 + 10.0) / 1100.0
        assert team.team_avg == pytest.approx(expected_avg)
        assert team.team_obp == pytest.approx(expected_obp)

    def test_multiple_teams(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Team A",
                    players=(_roster_player("y1", "B1", "B"),),
                ),
                TeamRoster(
                    team_key="t2",
                    team_name="Team B",
                    players=(_roster_player("y2", "B2", "B"),),
                ),
            ),
        )
        batting = [
            _make_batter_projection(player_id="fg1", hr=20.0),
            _make_batter_projection(player_id="fg2", hr=40.0),
        ]
        mapper = FakeIdMapper({"y1": "fg1", "y2": "fg2"})

        result = match_projections(rosters, batting, [], mapper)

        assert len(result) == 2
        hrs = {t.team_name: t.total_hr for t in result}
        assert hrs["Team A"] == pytest.approx(20.0)
        assert hrs["Team B"] == pytest.approx(40.0)

    def test_empty_roster_team(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(TeamRoster(team_key="t1", team_name="Empty", players=()),),
        )
        mapper = FakeIdMapper({})

        result = match_projections(rosters, [], [], mapper)

        team = result[0]
        assert team.total_hr == 0.0
        assert team.team_avg == 0.0
        assert team.team_era == 0.0
        assert team.unmatched_count == 0

    def test_pitcher_on_roster_not_in_projections(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Team A",
                    players=(_roster_player("yp1", "Pitcher", "P"),),
                ),
            ),
        )
        # Mapper has the ID but projection doesn't exist
        mapper = FakeIdMapper({"yp1": "fgp_missing"})

        result = match_projections(rosters, [], [], mapper)

        assert result[0].unmatched_count == 1

    def test_team_era_with_multiple_pitchers(self) -> None:
        rosters = LeagueRosters(
            league_key="lg1",
            teams=(
                TeamRoster(
                    team_key="t1",
                    team_name="Team A",
                    players=(
                        _roster_player("yp1", "P1", "P"),
                        _roster_player("yp2", "P2", "P"),
                    ),
                ),
            ),
        )
        pitching = [
            _make_pitcher_projection(player_id="fgp1", ip=200.0, er=40.0, h=160.0, bb=40.0, so=220.0),
            _make_pitcher_projection(player_id="fgp2", ip=100.0, er=50.0, h=100.0, bb=40.0, so=80.0),
        ]
        mapper = FakeIdMapper({"yp1": "fgp1", "yp2": "fgp2"})

        result = match_projections(rosters, [], pitching, mapper)

        team = result[0]
        expected_era = 90.0 / 300.0 * 9  # 2.70
        expected_whip = (260.0 + 80.0) / 300.0
        assert team.team_era == pytest.approx(expected_era)
        assert team.team_whip == pytest.approx(expected_whip)
        assert team.total_so == pytest.approx(300.0)
