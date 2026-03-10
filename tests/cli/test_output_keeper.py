from io import StringIO

import pytest
from rich.console import Console

import fantasy_baseball_manager.cli._output._keeper as _keeper_mod
from fantasy_baseball_manager.cli._output._keeper import print_keeper_draft_needs
from fantasy_baseball_manager.domain.category_tracker import (
    RosterAnalysis,
    TeamCategoryProjection,
)
from fantasy_baseball_manager.domain.keeper import (
    LeagueKeeperOverview,
    ProjectedKeeper,
    TeamKeeperProjection,
)


@pytest.fixture()
def _capture_console(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    buf = StringIO()
    monkeypatch.setattr(_keeper_mod, "console", Console(file=buf, highlight=False, no_color=True))
    return buf


def _user_team(keepers: tuple[ProjectedKeeper, ...] = ()) -> TeamKeeperProjection:
    return TeamKeeperProjection(
        team_key="1",
        team_name="My Team",
        is_user=True,
        keepers=keepers,
        total_value=sum(k.value for k in keepers),
        category_totals={},
    )


class TestPrintKeeperDraftNeeds:
    def test_no_keepers_shows_message(self, _capture_console: StringIO) -> None:
        overview = LeagueKeeperOverview(
            team_projections=(_user_team(),),
            trade_targets=(),
            category_names=(),
        )
        analysis = RosterAnalysis(projections=[], strongest_categories=[], weakest_categories=[])
        print_keeper_draft_needs(overview, analysis, [], 12)
        assert "No keepers projected" in _capture_console.getvalue()

    def test_with_keepers_and_projections(self, _capture_console: StringIO) -> None:
        keeper = ProjectedKeeper(
            player_id=1,
            player_name="Mike Trout",
            position="OF",
            value=30.0,
            category_scores={"HR": 5.0},
        )
        overview = LeagueKeeperOverview(
            team_projections=(_user_team(keepers=(keeper,)),),
            trade_targets=(),
            category_names=("HR",),
        )
        proj = TeamCategoryProjection(
            category="HR",
            projected_value=45.0,
            league_rank_estimate=3,
            strength="strong",
        )
        analysis = RosterAnalysis(
            projections=[proj],
            strongest_categories=["HR"],
            weakest_categories=[],
        )
        print_keeper_draft_needs(overview, analysis, [], 12)
        output = _capture_console.getvalue()
        assert "Mike Trout" in output
        assert "Category Strengths" in output
        assert "HR" in output
        assert "3/12" in output
