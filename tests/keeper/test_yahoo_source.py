from __future__ import annotations

import pytest

from fantasy_baseball_manager.keeper.yahoo_source import YahooKeeperData, YahooKeeperSource
from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamRoster


class FakeRosterSource:
    def __init__(self, rosters: LeagueRosters) -> None:
        self._rosters = rosters

    def fetch_rosters(self) -> LeagueRosters:
        return self._rosters


class FakeIdMapper:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._yahoo_to_fg = mapping

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return self._yahoo_to_fg.get(yahoo_id)

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return None

    def fangraphs_to_mlbam(self, fangraphs_id: str) -> str | None:
        return None

    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
        return None


def _make_player(yahoo_id: str, name: str, position_type: str, positions: tuple[str, ...]) -> RosterPlayer:
    return RosterPlayer(yahoo_id=yahoo_id, name=name, position_type=position_type, eligible_positions=positions)


def _make_rosters(user_team_key: str = "422.l.12345.t.1") -> LeagueRosters:
    user_team = TeamRoster(
        team_key="422.l.12345.t.1",
        team_name="My Team",
        players=(
            _make_player("100", "Batter A", "B", ("1B", "DH")),
            _make_player("101", "Pitcher B", "P", ("SP",)),
            _make_player("102", "Batter C", "B", ("OF", "DH")),
        ),
    )
    other_team_1 = TeamRoster(
        team_key="422.l.12345.t.2",
        team_name="Other Team 1",
        players=(
            _make_player("200", "Batter D", "B", ("SS",)),
            _make_player("201", "Pitcher E", "P", ("RP",)),
        ),
    )
    other_team_2 = TeamRoster(
        team_key="422.l.12345.t.3",
        team_name="Other Team 2",
        players=(
            _make_player("300", "Batter F", "B", ("2B", "3B")),
        ),
    )
    return LeagueRosters(league_key="422.l.12345", teams=(user_team, other_team_1, other_team_2))


def _make_id_mapping() -> dict[str, str]:
    return {
        "100": "fg100",
        "101": "fg101",
        "102": "fg102",
        "200": "fg200",
        "201": "fg201",
        # 300 intentionally missing to test unmapped
    }


class TestYahooKeeperSource:
    def test_splits_user_vs_other_teams(self) -> None:
        rosters = _make_rosters()
        mapper = FakeIdMapper(_make_id_mapping())
        source = YahooKeeperSource(
            roster_source=FakeRosterSource(rosters),
            id_mapper=mapper,
            user_team_key="422.l.12345.t.1",
        )
        data = source.fetch_keeper_data()

        assert set(data.user_candidate_ids) == {"fg100", "fg101", "fg102"}
        assert "fg200" in data.other_keeper_ids
        assert "fg201" in data.other_keeper_ids
        # fg100-102 should NOT be in other keepers
        assert "fg100" not in data.other_keeper_ids

    def test_maps_yahoo_ids_to_fangraphs(self) -> None:
        rosters = _make_rosters()
        mapper = FakeIdMapper({"100": "fg100", "101": "fg101", "102": "fg102", "200": "fg200", "201": "fg201"})
        source = YahooKeeperSource(
            roster_source=FakeRosterSource(rosters),
            id_mapper=mapper,
            user_team_key="422.l.12345.t.1",
        )
        data = source.fetch_keeper_data()

        assert "fg100" in data.user_candidate_ids
        assert "fg101" in data.user_candidate_ids

    def test_tracks_unmapped_ids(self) -> None:
        rosters = _make_rosters()
        # Only map user players + one other; "300" is unmapped
        mapper = FakeIdMapper(_make_id_mapping())
        source = YahooKeeperSource(
            roster_source=FakeRosterSource(rosters),
            id_mapper=mapper,
            user_team_key="422.l.12345.t.1",
        )
        data = source.fetch_keeper_data()

        assert "300" in data.unmapped_yahoo_ids

    def test_carries_eligible_positions(self) -> None:
        rosters = _make_rosters()
        mapper = FakeIdMapper(_make_id_mapping())
        source = YahooKeeperSource(
            roster_source=FakeRosterSource(rosters),
            id_mapper=mapper,
            user_team_key="422.l.12345.t.1",
        )
        data = source.fetch_keeper_data()

        assert data.user_candidate_positions["fg100"] == ("1B", "DH")
        assert data.user_candidate_positions["fg102"] == ("OF", "DH")
        assert data.user_candidate_positions["fg101"] == ("SP",)

    def test_raises_when_user_team_not_found(self) -> None:
        rosters = _make_rosters()
        mapper = FakeIdMapper(_make_id_mapping())
        source = YahooKeeperSource(
            roster_source=FakeRosterSource(rosters),
            id_mapper=mapper,
            user_team_key="422.l.99999.t.99",
        )

        with pytest.raises(ValueError, match="not found"):
            source.fetch_keeper_data()

    def test_unmapped_user_player_tracked(self) -> None:
        """User players that can't be mapped should appear in unmapped_yahoo_ids, not in candidates."""
        rosters = LeagueRosters(
            league_key="422.l.12345",
            teams=(
                TeamRoster(
                    team_key="422.l.12345.t.1",
                    team_name="My Team",
                    players=(
                        _make_player("999", "Unknown Guy", "B", ("OF",)),
                    ),
                ),
            ),
        )
        mapper = FakeIdMapper({})  # No mappings
        source = YahooKeeperSource(
            roster_source=FakeRosterSource(rosters),
            id_mapper=mapper,
            user_team_key="422.l.12345.t.1",
        )
        data = source.fetch_keeper_data()

        assert len(data.user_candidate_ids) == 0
        assert "999" in data.unmapped_yahoo_ids


class TestYahooKeeperDataFrozen:
    def test_is_frozen(self) -> None:
        data = YahooKeeperData(
            user_candidate_ids=("fg1",),
            user_candidate_positions={"fg1": ("1B",)},
            other_keeper_ids=frozenset({"fg2"}),
            unmapped_yahoo_ids=(),
        )
        assert data.user_candidate_ids == ("fg1",)
        assert data.other_keeper_ids == frozenset({"fg2"})
