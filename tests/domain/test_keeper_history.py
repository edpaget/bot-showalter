from fantasy_baseball_manager.domain.keeper_history import KeeperHistory, KeeperSeasonEntry


class TestKeeperSeasonEntry:
    def test_construction(self) -> None:
        entry = KeeperSeasonEntry(season=2025, cost=15.0, source="yahoo_draft")
        assert entry.season == 2025
        assert entry.cost == 15.0
        assert entry.source == "yahoo_draft"

    def test_frozen(self) -> None:
        entry = KeeperSeasonEntry(season=2025, cost=15.0, source="yahoo_draft")
        try:
            entry.season = 2026  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")  # noqa: TRY301
        except AttributeError:
            pass


class TestKeeperHistory:
    def test_construction(self) -> None:
        entries = (
            KeeperSeasonEntry(season=2024, cost=10.0, source="yahoo_draft"),
            KeeperSeasonEntry(season=2025, cost=12.0, source="yahoo_draft"),
        )
        history = KeeperHistory(player_id=1, player_name="Mike Trout", league="keeper", entries=entries)
        assert history.player_id == 1
        assert history.player_name == "Mike Trout"
        assert history.league == "keeper"
        assert len(history.entries) == 2

    def test_frozen(self) -> None:
        history = KeeperHistory(player_id=1, player_name="Mike Trout", league="keeper", entries=())
        try:
            history.player_id = 2  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")  # noqa: TRY301
        except AttributeError:
            pass

    def test_tuple_entries(self) -> None:
        entries = (KeeperSeasonEntry(season=2025, cost=15.0, source="yahoo_draft"),)
        history = KeeperHistory(player_id=1, player_name="Test", league="test", entries=entries)
        assert isinstance(history.entries, tuple)
