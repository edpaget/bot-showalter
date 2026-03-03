import datetime
from typing import Any, cast

from fantasy_baseball_manager.domain import Roster, RosterEntry, SeasonData, YahooDraftPick
from fantasy_baseball_manager.yahoo.history_source import YahooLeagueHistorySource


class FakeClient:
    def __init__(self, seasons: list[tuple[str, int]]) -> None:
        self._seasons = seasons

    def get_available_seasons(self) -> list[tuple[str, int]]:
        return self._seasons


class FakeDraftSource:
    def __init__(self, picks_by_key: dict[str, list[YahooDraftPick]] | None = None) -> None:
        self._picks_by_key = picks_by_key or {}

    def fetch_draft_results(self, league_key: str, season: int) -> list[YahooDraftPick]:
        return self._picks_by_key.get(league_key, [])


class FakeRosterSource:
    def __init__(self, rosters_by_key: dict[str, list[Roster]] | None = None) -> None:
        self._rosters_by_key = rosters_by_key or {}

    def fetch_all_rosters(
        self,
        *,
        team_keys: list[str],
        league_key: str,
        season: int,
        week: int,
        as_of: datetime.date,
    ) -> list[Roster]:
        return self._rosters_by_key.get(league_key, [])


def _make_source(
    client: Any = None,
    draft_source: Any = None,
    roster_source: Any = None,
) -> YahooLeagueHistorySource:
    return YahooLeagueHistorySource(
        client=cast("Any", client or FakeClient(seasons=[])),
        draft_source=cast("Any", draft_source or FakeDraftSource()),
        roster_source=cast("Any", roster_source or FakeRosterSource()),
    )


class TestDiscoverSeasons:
    def test_constructs_league_keys(self) -> None:
        client = FakeClient(seasons=[("449", 2026), ("422", 2025)])
        source = _make_source(client=client)

        result = source.discover_seasons(league_id=12345)

        assert result == [("449.l.12345", 2026), ("422.l.12345", 2025)]

    def test_empty_seasons(self) -> None:
        source = _make_source()

        result = source.discover_seasons(league_id=12345)

        assert result == []


class TestFetchSeasonData:
    def test_delegates_to_sources(self) -> None:
        pick = YahooDraftPick(
            league_key="449.l.12345",
            season=2026,
            round=1,
            pick=1,
            team_key="449.l.12345.t.1",
            yahoo_player_key="449.p.1234",
            player_id=100,
            player_name="Mike Trout",
            position="OF",
            cost=45,
        )
        roster = Roster(
            team_key="449.l.12345.t.1",
            league_key="449.l.12345",
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 1),
            entries=(
                RosterEntry(
                    player_id=100,
                    yahoo_player_key="449.p.1234",
                    player_name="Mike Trout",
                    position="OF",
                    roster_status="active",
                    acquisition_type="draft",
                ),
            ),
        )
        draft_source = FakeDraftSource(picks_by_key={"449.l.12345": [pick]})
        roster_source = FakeRosterSource(rosters_by_key={"449.l.12345": [roster]})
        client = FakeClient(seasons=[])

        source = _make_source(client=client, draft_source=draft_source, roster_source=roster_source)

        result = source.fetch_season_data(
            league_key="449.l.12345",
            season=2026,
            team_keys=["449.l.12345.t.1"],
            as_of=datetime.date(2026, 3, 1),
        )

        assert isinstance(result, SeasonData)
        assert result.league_key == "449.l.12345"
        assert result.season == 2026
        assert len(result.draft_picks) == 1
        assert result.draft_picks[0].player_name == "Mike Trout"
        assert len(result.rosters) == 1

    def test_empty_draft(self) -> None:
        source = _make_source()

        result = source.fetch_season_data(
            league_key="449.l.12345",
            season=2026,
            team_keys=["449.l.12345.t.1"],
            as_of=datetime.date(2026, 3, 1),
        )

        assert result.draft_picks == []
        assert result.rosters == []
