"""Tests for _build_player_teams in cli/commands/ingest.py."""

from typing import TYPE_CHECKING

from fantasy_baseball_manager.cli.commands.ingest import _build_player_teams

if TYPE_CHECKING:
    import pytest
from fantasy_baseball_manager.cli.factory import IngestContainer
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain import Player, RosterStint, Team


def _player_id(container: IngestContainer, mlbam_id: int) -> int:
    """Look up a player's internal id by mlbam_id, asserting it exists."""
    pid = next(p.id for p in container.player_repo.all() if p.mlbam_id == mlbam_id)
    assert pid is not None
    return pid


def _seed(container: IngestContainer) -> None:
    """Seed two teams, two players, and roster stints for 2024."""
    conn = container.conn
    container.team_repo.upsert(Team(abbreviation="NYY", name="Yankees", league="AL", division="East"))
    container.team_repo.upsert(Team(abbreviation="BOS", name="Red Sox", league="AL", division="East"))
    conn.commit()

    teams = {t.abbreviation: t.id for t in container.team_repo.all() if t.id is not None}
    nyy_id = teams["NYY"]
    bos_id = teams["BOS"]

    container.player_repo.upsert(Player(name_first="Aaron", name_last="Judge", mlbam_id=592450))
    container.player_repo.upsert(Player(name_first="Rafael", name_last="Devers", mlbam_id=646240))
    conn.commit()

    judge_id = _player_id(container, 592450)
    devers_id = _player_id(container, 646240)

    container.roster_stint_repo.upsert(
        RosterStint(player_id=judge_id, team_id=nyy_id, season=2024, start_date="2024-03-28")
    )
    container.roster_stint_repo.upsert(
        RosterStint(player_id=devers_id, team_id=bos_id, season=2024, start_date="2024-03-28")
    )
    conn.commit()


class TestBuildPlayerTeams:
    def test_lahman_stints_build_mapping(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        _seed(container)

        result = _build_player_teams(container, 2024)

        judge_id = _player_id(container, 592450)
        devers_id = _player_id(container, 646240)
        assert result[judge_id] == "NYY"
        assert result[devers_id] == "BOS"
        conn.close()

    def test_falls_back_to_prior_season(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        _seed(container)  # seeds stints for 2024

        # Request 2025 — no stints for 2025, should fall back to 2024
        result = _build_player_teams(container, 2025)

        judge_id = _player_id(container, 592450)
        assert result[judge_id] == "NYY"
        conn.close()

    def test_mlb_api_overlays_updates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        _seed(container)

        judge_id = _player_id(container, 592450)

        # MLB API says Judge moved to LAD
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.ingest.fetch_mlb_active_teams",
            lambda _season: {592450: "LAD"},
        )

        result = _build_player_teams(container, 2024)

        assert result[judge_id] == "LAD"
        conn.close()

    def test_mlb_api_failure_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        _seed(container)

        def _raise(_season: int) -> dict[int, str]:
            msg = "network error"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.ingest.fetch_mlb_active_teams",
            _raise,
        )

        result = _build_player_teams(container, 2024)

        # Should still have Lahman data
        judge_id = _player_id(container, 592450)
        assert result[judge_id] == "NYY"
        conn.close()
