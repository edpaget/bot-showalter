from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.adp.models import ADPData, ADPEntry
from fantasy_baseball_manager.cli import app
from fantasy_baseball_manager.config import create_config
from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.services import ServiceContainer, get_container, set_container

runner = CliRunner()

YEARS = [2024, 2023, 2022]


def _make_batter(
    player_id: str = "b1",
    name: str = "Test Hitter",
    year: int = 2024,
    age: int = 28,
    pa: int = 600,
    ab: int = 540,
    h: int = 160,
    singles: int = 100,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 25,
    bb: int = 50,
    so: int = 120,
    hbp: int = 5,
    sf: int = 3,
    sh: int = 2,
    sb: int = 10,
    cs: int = 3,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        pa=pa,
        ab=ab,
        h=h,
        singles=singles,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
        sh=sh,
        sb=sb,
        cs=cs,
        r=80,
        rbi=90,
    )


def _make_pitcher(
    player_id: str = "p1",
    name: str = "Test Pitcher",
    year: int = 2024,
    age: int = 28,
    ip: float = 180.0,
    g: int = 32,
    gs: int = 32,
    er: int = 70,
    h: int = 150,
    bb: int = 50,
    so: int = 200,
    hr: int = 20,
    hbp: int = 5,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        ip=ip,
        g=g,
        gs=gs,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=hr,
        hbp=hbp,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


def _make_league_batting(year: int = 2024) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        pa=6000,
        ab=5400,
        h=1500,
        singles=900,
        doubles=300,
        triples=30,
        hr=200,
        bb=500,
        so=1400,
        hbp=50,
        sf=30,
        sh=20,
        sb=100,
        cs=30,
        r=800,
        rbi=750,
    )


def _make_league_pitching(year: int = 2024) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        ip=1400.0,
        g=500,
        gs=200,
        er=600,
        h=1300,
        bb=450,
        so=1300,
        hr=180,
        hbp=50,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


class FakeDataSource:
    def __init__(
        self,
        player_batting: dict[int, list[BattingSeasonStats]],
        player_pitching: dict[int, list[PitchingSeasonStats]],
        team_batting_stats: dict[int, list[BattingSeasonStats]],
        team_pitching_stats: dict[int, list[PitchingSeasonStats]],
    ) -> None:
        self._player_batting = player_batting
        self._player_pitching = player_pitching
        self._team_batting = team_batting_stats
        self._team_pitching = team_pitching_stats

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._player_batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._player_pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return self._team_batting.get(year, [])

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._team_pitching.get(year, [])


def _build_fake(
    batting: bool = True,
    pitching: bool = True,
    num_batters: int = 3,
    num_pitchers: int = 3,
) -> FakeDataSource:
    batter_configs = [
        ("b1", "Slugger Jones", 40, 5, 600),
        ("b2", "Speedy Smith", 10, 30, 600),
        ("b3", "Average Andy", 20, 15, 600),
    ]
    pitcher_configs = [
        ("p1", "Ace Adams", 250, 50, 130),
        ("p2", "Bullpen Bob", 150, 70, 180),
        ("p3", "Middle Mike", 180, 60, 155),
    ]

    player_batting: dict[int, list[BattingSeasonStats]] = {}
    team_batting: dict[int, list[BattingSeasonStats]] = {}
    player_pitching: dict[int, list[PitchingSeasonStats]] = {}
    team_pitching: dict[int, list[PitchingSeasonStats]] = {}

    if batting:
        for y in YEARS:
            batters: list[BattingSeasonStats] = []
            for i in range(min(num_batters, len(batter_configs))):
                pid, name, hr, sb, pa = batter_configs[i]
                batters.append(
                    _make_batter(
                        player_id=pid,
                        name=name,
                        year=y,
                        age=28 - (2024 - y),
                        hr=hr,
                        sb=sb,
                        pa=pa,
                    )
                )
            player_batting[y] = batters
            team_batting[y] = [_make_league_batting(year=y)]

    if pitching:
        for y in YEARS:
            pitchers: list[PitchingSeasonStats] = []
            for i in range(min(num_pitchers, len(pitcher_configs))):
                pid, name, so, er, h = pitcher_configs[i]
                pitchers.append(
                    _make_pitcher(
                        player_id=pid,
                        name=name,
                        year=y,
                        age=28 - (2024 - y),
                        so=so,
                        er=er,
                        h=h,
                    )
                )
            player_pitching[y] = pitchers
            team_pitching[y] = [_make_league_pitching(year=y)]

    return FakeDataSource(
        player_batting=player_batting,
        player_pitching=player_pitching,
        team_batting_stats=team_batting,
        team_pitching_stats=team_pitching,
    )


@pytest.fixture(autouse=True)
def reset_container() -> Generator[None]:
    yield
    set_container(None)


def _install_fake(batting: bool = True, pitching: bool = True) -> None:
    ds = _build_fake(batting=batting, pitching=pitching)
    set_container(ServiceContainer(data_source=ds))


class TestDraftRankCommand:
    def test_default_shows_rankings(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025"])
        assert result.exit_code == 0
        assert "Draft rankings" in result.output

    def test_batting_only(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--batting"])
        assert result.exit_code == 0
        assert "Slugger Jones" in result.output
        assert "Ace Adams" not in result.output

    def test_pitching_only(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--pitching"])
        assert result.exit_code == 0
        assert "Ace Adams" in result.output
        assert "Slugger Jones" not in result.output

    def test_top_limits_output(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--batting", "--top", "2"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        # header + separator + 2 data lines = at least 4 lines total (plus title)
        data_lines = [line for line in lines if any(name in line for name in ["Slugger", "Speedy", "Average"])]
        assert len(data_lines) == 2

    def test_weight_option(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--batting", "--weight", "HR=2.0"])
        assert result.exit_code == 0
        assert "Slugger Jones" in result.output

    def test_invalid_weight_format(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--weight", "INVALID"])
        assert result.exit_code == 1

    def test_unknown_engine_rejected(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--engine", "steamer"])
        assert result.exit_code == 1

    def test_positions_file(self, tmp_path: Path) -> None:
        _install_fake(pitching=False)
        pos_file = tmp_path / "positions.csv"
        pos_file.write_text("b1,1B/OF\nb2,OF\nb3,SS\n")
        result = runner.invoke(
            app,
            ["players", "draft-rank", "2025", "--batting", "--positions", str(pos_file)],
        )
        assert result.exit_code == 0
        assert "1B/OF" in result.output or "OF" in result.output

    def test_roster_config_file(self, tmp_path: Path) -> None:
        _install_fake(pitching=False)
        config_file = tmp_path / "roster.yaml"
        config_file.write_text("slots:\n  C: 1\n  OF: 3\n  SP: 5\n  BN: 2\n")
        result = runner.invoke(
            app,
            ["players", "draft-rank", "2025", "--batting", "--roster-config", str(config_file)],
        )
        assert result.exit_code == 0

    def test_yahoo_fetches_positions_and_draft_results(self) -> None:
        from unittest.mock import MagicMock

        ds = _build_fake(pitching=False)

        league = MagicMock()
        league.taken_players.return_value = [
            {"player_id": 101, "name": "Slugger Jones", "eligible_positions": ["1B", "DH"], "position_type": "B"},
            {"player_id": 102, "name": "Speedy Smith", "eligible_positions": ["OF"], "position_type": "B"},
            {"player_id": 103, "name": "Average Andy", "eligible_positions": ["SS"], "position_type": "B"},
        ]
        league.free_agents.return_value = []
        league.draft_results.return_value = [
            {"player_key": "422.p.101", "player_id": "101", "team_key": "422.l.1.t.2", "round": 1, "pick": 1},
        ]
        league.settings.return_value = {"draft_status": "postdraft"}
        league.team_key.return_value = "422.l.1.t.1"

        mapper = MagicMock()
        mapper.yahoo_to_fangraphs.side_effect = lambda yid: {
            "101": "b1",
            "102": "b2",
            "103": "b3",
        }.get(yid)

        set_container(ServiceContainer(data_source=ds, yahoo_league=league, id_mapper=mapper))
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--batting", "--yahoo"])
        assert result.exit_code == 0
        # b1 was drafted by another team — should be excluded
        assert "Slugger Jones" not in result.output
        # b2 and b3 not drafted — should appear
        assert "Speedy Smith" in result.output
        assert "Average Andy" in result.output

    def test_yahoo_identifies_user_picks(self) -> None:
        from unittest.mock import MagicMock

        ds = _build_fake(pitching=False)

        league = MagicMock()
        league.taken_players.return_value = [
            {"player_id": 101, "name": "Slugger Jones", "eligible_positions": ["1B"], "position_type": "B"},
            {"player_id": 102, "name": "Speedy Smith", "eligible_positions": ["OF"], "position_type": "B"},
        ]
        league.free_agents.return_value = []
        league.draft_results.return_value = [
            {"player_key": "422.p.101", "player_id": "101", "team_key": "422.l.1.t.1", "round": 1, "pick": 1},
        ]
        league.settings.return_value = {"draft_status": "postdraft"}
        league.team_key.return_value = "422.l.1.t.1"

        mapper = MagicMock()
        mapper.yahoo_to_fangraphs.side_effect = lambda yid: {"101": "b1", "102": "b2"}.get(yid)

        set_container(ServiceContainer(data_source=ds, yahoo_league=league, id_mapper=mapper))
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--batting", "--yahoo"])
        assert result.exit_code == 0
        # b1 was drafted by user's team — should be excluded from rankings
        assert "Slugger Jones" not in result.output

    def test_accepts_league_id(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--league-id", "99999"])
        assert result.exit_code == 0

    def test_accepts_season(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025", "--season", "2024"])
        assert result.exit_code == 0

    def test_overrides_cleared_after_draft_rank(self) -> None:
        _install_fake()
        runner.invoke(app, ["players", "draft-rank", "2025", "--league-id", "99999"])
        cfg = create_config(yaml_path="/nonexistent/config.yaml")
        assert cfg["league.id"] == ""

    def test_util_hidden_when_other_positions_exist(self, tmp_path: Path) -> None:
        _install_fake(pitching=False)
        pos_file = tmp_path / "positions.csv"
        pos_file.write_text("b1,1B/DH\nb2,OF\nb3,SS\n")
        result = runner.invoke(
            app,
            ["players", "draft-rank", "2025", "--batting", "--positions", str(pos_file)],
        )
        assert result.exit_code == 0
        # DH normalizes to Util; Util should be hidden when 1B is also present
        assert "Util" not in result.output
        assert "1B" in result.output

    def test_util_shown_when_only_position(self, tmp_path: Path) -> None:
        _install_fake(pitching=False)
        pos_file = tmp_path / "positions.csv"
        pos_file.write_text("b1,DH\nb2,OF\nb3,SS\n")
        result = runner.invoke(
            app,
            ["players", "draft-rank", "2025", "--batting", "--positions", str(pos_file)],
        )
        assert result.exit_code == 0
        assert "Util" in result.output

    def test_generic_p_excluded_from_batter_positions(self, tmp_path: Path) -> None:
        """Yahoo returns 'P' for two-way players; it should not appear in batter positions."""
        # Give b1 both batter and pitcher projections by reusing ID in both
        ds = _build_fake(batting=True, pitching=True, num_batters=1, num_pitchers=1)
        # Override so the same player ID appears in both batting and pitching
        for year_batters in ds._player_batting.values():
            for b in year_batters:
                object.__setattr__(b, "player_id", "two_way")
        for year_pitchers in ds._player_pitching.values():
            for p in year_pitchers:
                object.__setattr__(p, "player_id", "two_way")
                object.__setattr__(p, "name", "Two Way Player")
        set_container(ServiceContainer(data_source=ds))

        pos_file = tmp_path / "positions.csv"
        # Simulate Yahoo-like positions: Util (from DH) + P (generic pitcher)
        pos_file.write_text("two_way,Util/P\n")
        result = runner.invoke(
            app,
            ["players", "draft-rank", "2025", "--positions", str(pos_file)],
        )
        assert result.exit_code == 0
        # The batter entry should not show "P" in positions
        for line in result.output.split("\n"):
            if "Two Way Player" in line and "SP" not in line and "RP" not in line:
                # This is the batter line — should not contain "/P" or standalone "P"
                assert "/P" not in line

    def test_best_column_removed_from_header(self) -> None:
        _install_fake()
        result = runner.invoke(app, ["players", "draft-rank", "2025"])
        assert result.exit_code == 0
        assert "Best" not in result.output

    def test_positions_file_takes_precedence_over_yahoo(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        ds = _build_fake(pitching=False)

        league = MagicMock()
        league.draft_results.return_value = []
        league.settings.return_value = {"draft_status": "predraft"}
        league.team_key.return_value = "422.l.1.t.1"

        set_container(ServiceContainer(data_source=ds, yahoo_league=league, id_mapper=MagicMock()))

        pos_file = tmp_path / "positions.csv"
        pos_file.write_text("b1,SS\n")
        result = runner.invoke(
            app,
            ["players", "draft-rank", "2025", "--batting", "--positions", str(pos_file), "--yahoo"],
        )
        assert result.exit_code == 0
        # --positions file should be used for positions, not Yahoo
        league.taken_players.assert_not_called()

    def test_adp_flag_invokes_adp_source(self) -> None:
        """Test that --adp flag fetches and displays ADP data."""
        _install_fake(pitching=False)

        mock_adp_data = ADPData(
            entries=(
                ADPEntry(name="Slugger Jones", adp=5.0, positions=("1B",)),
                ADPEntry(name="Speedy Smith", adp=15.0, positions=("OF",)),
            ),
            fetched_at=datetime.now(UTC),
        )

        with patch(
            "fantasy_baseball_manager.draft.cli.YahooADPScraper"
        ) as mock_scraper_cls:
            mock_scraper = MagicMock()
            mock_scraper.fetch_adp.return_value = mock_adp_data
            mock_scraper_cls.return_value = mock_scraper

            result = runner.invoke(app, ["players", "draft-rank", "2025", "--batting", "--adp"])

            assert result.exit_code == 0
            assert "ADP" in result.output
            assert "Diff" in result.output
            mock_scraper.fetch_adp.assert_called_once()

    def test_no_adp_cache_invalidates_and_fetches_fresh(self) -> None:
        """Test that --no-adp-cache invalidates cache and fetches fresh data."""
        _install_fake(pitching=False)
        container = get_container()

        mock_adp_data = ADPData(
            entries=(ADPEntry(name="Slugger Jones", adp=5.0, positions=("1B",)),),
            fetched_at=datetime.now(UTC),
        )

        with patch(
            "fantasy_baseball_manager.draft.cli.YahooADPScraper"
        ) as mock_scraper_cls, patch(
            "fantasy_baseball_manager.draft.cli.CachedADPSource"
        ) as mock_cached_cls, patch.object(
            container.cache_store, "invalidate"
        ) as mock_invalidate:
            mock_scraper = MagicMock()
            mock_scraper_cls.return_value = mock_scraper

            mock_cached = MagicMock()
            mock_cached.fetch_adp.return_value = mock_adp_data
            mock_cached_cls.return_value = mock_cached

            result = runner.invoke(
                app, ["players", "draft-rank", "2025", "--batting", "--adp", "--no-adp-cache"]
            )

            assert result.exit_code == 0
            # Cache should be invalidated before fetch
            mock_invalidate.assert_called_once_with("adp_data", "yahoo_v2")
            # CachedADPSource should still be used (to write fresh data)
            mock_cached_cls.assert_called_once()

    def test_adp_uses_cache_by_default(self) -> None:
        """Test that --adp uses cache by default."""
        _install_fake(pitching=False)

        mock_adp_data = ADPData(
            entries=(ADPEntry(name="Slugger Jones", adp=5.0, positions=("1B",)),),
            fetched_at=datetime.now(UTC),
        )

        with patch(
            "fantasy_baseball_manager.draft.cli.YahooADPScraper"
        ) as mock_scraper_cls, patch(
            "fantasy_baseball_manager.draft.cli.CachedADPSource"
        ) as mock_cached_cls:
            mock_scraper = MagicMock()
            mock_scraper_cls.return_value = mock_scraper

            mock_cached = MagicMock()
            mock_cached.fetch_adp.return_value = mock_adp_data
            mock_cached_cls.return_value = mock_cached

            result = runner.invoke(app, ["players", "draft-rank", "2025", "--batting", "--adp"])

            assert result.exit_code == 0
            # CachedADPSource should wrap the scraper
            mock_cached_cls.assert_called_once()
            mock_cached.fetch_adp.assert_called_once()
