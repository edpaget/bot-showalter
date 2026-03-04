"""Tests for _build_cohort_assignments in cli/commands/standalone.py."""

import pytest

from fantasy_baseball_manager.cli.commands.standalone import _build_cohort_assignments
from fantasy_baseball_manager.cli.factory import EvalContext
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain import BattingStats, Player
from fantasy_baseball_manager.repos import (
    SqliteBattingStatsRepo,
    SqlitePitchingStatsRepo,
    SqlitePlayerRepo,
    SqliteProjectionRepo,
)
from fantasy_baseball_manager.services import ProjectionEvaluator


def _make_eval_context() -> EvalContext:
    """Build an EvalContext backed by an in-memory database."""
    conn = create_connection(":memory:")
    player_repo = SqlitePlayerRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    projection_repo = SqliteProjectionRepo(conn)
    pitching_repo = SqlitePitchingStatsRepo(conn)
    evaluator = ProjectionEvaluator(projection_repo, batting_repo, pitching_repo)
    return EvalContext(
        conn=conn,
        evaluator=evaluator,
        player_repo=player_repo,
        batting_repo=batting_repo,
        projection_repo=projection_repo,
    )


class TestBuildCohortAssignments:
    def test_age_dimension(self) -> None:
        ctx = _make_eval_context()
        # Young player (born 2000) → age 24 in 2024
        pid = ctx.player_repo.upsert(Player(name_first="Young", name_last="Player", birth_date="2000-01-15"))
        ctx.conn.commit()

        result = _build_cohort_assignments(ctx, "age", 2024)

        assert pid in result
        # 24 years old → "young" cohort
        assert result[pid] == "young"
        ctx.conn.close()

    def test_experience_dimension(self) -> None:
        ctx = _make_eval_context()
        pid = ctx.player_repo.upsert(Player(name_first="Rookie", name_last="Player"))
        ctx.conn.commit()

        # Seed current-season actuals so this player is in the player_ids set
        ctx.batting_repo.upsert(BattingStats(player_id=pid, season=2024, source="fangraphs", pa=100))
        ctx.conn.commit()

        # No prior batting → "rookie"
        result = _build_cohort_assignments(ctx, "experience", 2024)

        assert pid in result
        assert result[pid] == "rookie"
        ctx.conn.close()

    def test_top300_dimension(self) -> None:
        ctx = _make_eval_context()
        pid = ctx.player_repo.upsert(Player(name_first="Star", name_last="Player"))
        ctx.conn.commit()

        # Single player with WAR → top300
        ctx.batting_repo.upsert(BattingStats(player_id=pid, season=2024, source="fangraphs", pa=500))
        ctx.conn.commit()

        result = _build_cohort_assignments(ctx, "top300", 2024)

        assert pid in result
        ctx.conn.close()

    def test_unknown_dimension_raises(self) -> None:
        ctx = _make_eval_context()

        with pytest.raises(ValueError, match="unknown dimension"):
            _build_cohort_assignments(ctx, "nonsense", 2024)

        ctx.conn.close()
