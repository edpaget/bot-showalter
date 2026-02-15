from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.il_stint_repo import SqliteILStintRepo
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.repos.statcast_pitch_repo import SqliteStatcastPitchRepo
from fantasy_baseball_manager.repos.protocols import (
    BattingStatsRepo,
    ILStintRepo,
    LoadLogRepo,
    ModelRunRepo,
    PitchingStatsRepo,
    PlayerRepo,
    PositionAppearanceRepo,
    ProjectionRepo,
    RosterStintRepo,
    StatcastPitchRepo,
    TeamRepo,
)


class TestProtocolConformance:
    def test_player_repo_conforms(self) -> None:
        assert isinstance(SqlitePlayerRepo, type)
        assert issubclass(SqlitePlayerRepo, PlayerRepo)

    def test_team_repo_conforms(self) -> None:
        assert issubclass(SqliteTeamRepo, TeamRepo)

    def test_batting_stats_repo_conforms(self) -> None:
        assert issubclass(SqliteBattingStatsRepo, BattingStatsRepo)

    def test_pitching_stats_repo_conforms(self) -> None:
        assert issubclass(SqlitePitchingStatsRepo, PitchingStatsRepo)

    def test_projection_repo_conforms(self) -> None:
        assert issubclass(SqliteProjectionRepo, ProjectionRepo)

    def test_statcast_pitch_repo_conforms(self) -> None:
        assert issubclass(SqliteStatcastPitchRepo, StatcastPitchRepo)

    def test_model_run_repo_conforms(self) -> None:
        assert issubclass(SqliteModelRunRepo, ModelRunRepo)

    def test_il_stint_repo_conforms(self) -> None:
        assert issubclass(SqliteILStintRepo, ILStintRepo)

    def test_position_appearance_repo_conforms(self) -> None:
        assert issubclass(SqlitePositionAppearanceRepo, PositionAppearanceRepo)

    def test_roster_stint_repo_conforms(self) -> None:
        assert issubclass(SqliteRosterStintRepo, RosterStintRepo)

    def test_load_log_repo_conforms(self) -> None:
        assert issubclass(SqliteLoadLogRepo, LoadLogRepo)
