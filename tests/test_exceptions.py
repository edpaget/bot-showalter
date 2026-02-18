from fantasy_baseball_manager.config_league import LeagueConfigError
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.exceptions import FbmException
from fantasy_baseball_manager.repos.errors import PlayerConflictError


class TestFbmException:
    def test_is_exception(self) -> None:
        assert issubclass(FbmException, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        try:
            raise FbmException("test")
        except FbmException as e:
            assert str(e) == "test"


class TestExceptionInheritance:
    def test_league_config_error_inherits_fbm_exception(self) -> None:
        assert issubclass(LeagueConfigError, FbmException)
        assert issubclass(LeagueConfigError, Exception)

    def test_player_conflict_error_inherits_fbm_exception(self) -> None:
        assert issubclass(PlayerConflictError, FbmException)
        assert issubclass(PlayerConflictError, Exception)

    def test_league_config_error_still_caught_as_exception(self) -> None:
        try:
            raise LeagueConfigError("config bad")
        except Exception as e:
            assert "config bad" in str(e)

    def test_player_conflict_error_still_caught_as_exception(self) -> None:
        p = Player(name_first="A", name_last="B", mlbam_id=1)
        try:
            raise PlayerConflictError(p, p, "mlbam_id")
        except Exception:
            pass
