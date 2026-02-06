"""Tests for the service container."""

from fantasy_baseball_manager.player_id.mapper import SfbbMapper
from fantasy_baseball_manager.services import (
    ServiceConfig,
    ServiceContainer,
    get_container,
    set_container,
)


class FakeBattingSource:
    """Fake DataSource for testing."""

    pass


class FakeRosterSource:
    """Fake roster source for testing."""

    pass


class FakeBlender:
    """Fake blender for testing."""

    pass


class FakeYahooLeague:
    """Fake Yahoo league for testing."""

    pass


class TestServiceConfig:
    def test_default_no_cache_is_false(self) -> None:
        config = ServiceConfig()
        assert config.no_cache is False

    def test_no_cache_can_be_set_true(self) -> None:
        config = ServiceConfig(no_cache=True)
        assert config.no_cache is True


class TestServiceContainer:
    def test_injected_batting_source_is_returned(self) -> None:
        fake = FakeBattingSource()
        container = ServiceContainer(batting_source=fake)  # type: ignore[arg-type]
        assert container.batting_source is fake

    def test_injected_id_mapper_is_returned(self) -> None:
        fake = SfbbMapper({}, {})
        container = ServiceContainer(id_mapper=fake)
        assert container.id_mapper is fake

    def test_injected_roster_source_is_returned(self) -> None:
        fake = FakeRosterSource()
        container = ServiceContainer(roster_source=fake)  # type: ignore[arg-type]
        assert container.roster_source is fake

    def test_injected_blender_is_returned(self) -> None:
        fake = FakeBlender()
        container = ServiceContainer(blender=fake)  # type: ignore[arg-type]
        assert container.blender is fake

    def test_injected_yahoo_league_is_returned(self) -> None:
        fake = FakeYahooLeague()
        container = ServiceContainer(yahoo_league=fake)
        assert container.yahoo_league is fake

    def test_config_defaults_when_not_provided(self) -> None:
        container = ServiceContainer()
        assert container.config.no_cache is False

    def test_config_is_returned(self) -> None:
        config = ServiceConfig(no_cache=True)
        container = ServiceContainer(config=config)
        assert container.config.no_cache is True

    def test_batting_source_is_cached(self) -> None:
        fake = FakeBattingSource()
        container = ServiceContainer(batting_source=fake)  # type: ignore[arg-type]
        assert container.batting_source is container.batting_source


class TestGetSetContainer:
    def test_get_container_returns_same_instance(self) -> None:
        set_container(None)  # Reset
        c1 = get_container()
        c2 = get_container()
        assert c1 is c2

    def test_set_container_replaces_instance(self) -> None:
        set_container(None)  # Reset
        fake_ds = FakeBattingSource()
        custom = ServiceContainer(batting_source=fake_ds)  # type: ignore[arg-type]
        set_container(custom)
        assert get_container() is custom
        assert get_container().batting_source is fake_ds

    def test_set_container_none_resets(self) -> None:
        fake_ds = FakeBattingSource()
        custom = ServiceContainer(batting_source=fake_ds)  # type: ignore[arg-type]
        set_container(custom)
        set_container(None)
        # Next get should create a new default container
        new_container = get_container()
        assert new_container is not custom

    def teardown_method(self) -> None:
        # Reset after each test
        set_container(None)
