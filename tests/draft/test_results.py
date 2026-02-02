from __future__ import annotations

from unittest.mock import MagicMock

from fantasy_baseball_manager.draft.results import (
    DraftStatus,
    YahooDraftPick,
    YahooDraftResultsSource,
)


class TestYahooDraftResultsSource:
    def test_maps_draft_results_to_picks(self) -> None:
        league = MagicMock()
        league.draft_results.return_value = [
            {
                "player_key": "422.p.10660",
                "player_id": "10660",
                "team_key": "422.l.12345.t.1",
                "round": 1,
                "pick": 1,
            },
            {
                "player_key": "422.p.9542",
                "player_id": "9542",
                "team_key": "422.l.12345.t.2",
                "round": 1,
                "pick": 2,
            },
        ]

        source = YahooDraftResultsSource(league)
        picks = source.fetch_draft_results()

        assert len(picks) == 2
        assert picks[0] == YahooDraftPick(player_id="10660", team_key="422.l.12345.t.1", round=1, pick=1)
        assert picks[1] == YahooDraftPick(player_id="9542", team_key="422.l.12345.t.2", round=1, pick=2)

    def test_empty_draft_results(self) -> None:
        league = MagicMock()
        league.draft_results.return_value = []

        source = YahooDraftResultsSource(league)
        picks = source.fetch_draft_results()

        assert picks == []

    def test_draft_status_pre_draft(self) -> None:
        league = MagicMock()
        league.settings.return_value = {"draft_status": "predraft"}

        source = YahooDraftResultsSource(league)
        status = source.fetch_draft_status()

        assert status == DraftStatus.PRE_DRAFT

    def test_draft_status_in_progress(self) -> None:
        league = MagicMock()
        league.settings.return_value = {"draft_status": "draft"}

        source = YahooDraftResultsSource(league)
        status = source.fetch_draft_status()

        assert status == DraftStatus.IN_PROGRESS

    def test_draft_status_post_draft(self) -> None:
        league = MagicMock()
        league.settings.return_value = {"draft_status": "postdraft"}

        source = YahooDraftResultsSource(league)
        status = source.fetch_draft_status()

        assert status == DraftStatus.POST_DRAFT

    def test_user_team_key(self) -> None:
        league = MagicMock()
        league.team_key.return_value = "422.l.12345.t.3"

        source = YahooDraftResultsSource(league)
        key = source.fetch_user_team_key()

        assert key == "422.l.12345.t.3"

    def test_picks_are_frozen(self) -> None:
        pick = YahooDraftPick(player_id="10660", team_key="422.l.12345.t.1", round=1, pick=1)
        assert pick.player_id == "10660"
        # Frozen dataclass — attribute assignment should raise
        try:
            pick.player_id = "other"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass
