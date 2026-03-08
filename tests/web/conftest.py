import pytest
from fastapi.testclient import TestClient

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    Player,
    StatType,
    Valuation,
)
from fantasy_baseball_manager.repos import SqlitePlayerRepo, SqliteValuationRepo
from fantasy_baseball_manager.web import create_app

_LEAGUE = LeagueSettings(
    name="Test League",
    format=LeagueFormat.H2H_CATEGORIES,
    teams=10,
    budget=260,
    roster_batters=14,
    roster_pitchers=9,
    batting_categories=(
        CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        CategoryConfig(key="RBI", name="Runs Batted In", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
    ),
    pitching_categories=(
        CategoryConfig(key="W", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
    ),
    positions={"C": 1, "1B": 1, "OF": 3},
    pitcher_positions={"SP": 5, "RP": 2},
)


def _seed_data(provider: SingleConnectionProvider) -> None:
    """Seed the in-memory database with test players and valuations."""
    player_repo = SqlitePlayerRepo(provider)
    valuation_repo = SqliteValuationRepo(provider)

    players = [
        Player(name_first="Mike", name_last="Trout", id=1, mlbam_id=545361),
        Player(name_first="Shohei", name_last="Ohtani", id=2, mlbam_id=660271),
        Player(name_first="Gerrit", name_last="Cole", id=3, mlbam_id=543037),
    ]
    for p in players:
        player_repo.upsert(p)

    valuations = [
        Valuation(
            player_id=1,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026",
            player_type="batter",
            position="OF",
            value=35.0,
            rank=1,
            category_scores={"HR": 2.5, "RBI": 1.8},
        ),
        Valuation(
            player_id=2,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026",
            player_type="batter",
            position="OF",
            value=30.0,
            rank=2,
            category_scores={"HR": 2.0, "RBI": 1.5},
        ),
        Valuation(
            player_id=3,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="2026",
            player_type="pitcher",
            position="SP",
            value=25.0,
            rank=3,
            category_scores={"W": 1.5, "K": 2.0},
        ),
    ]
    for v in valuations:
        valuation_repo.upsert(v)

    provider._conn.commit()


@pytest.fixture
def client() -> TestClient:
    """Create a test client with an in-memory SQLite database seeded with test data."""
    conn = create_connection(":memory:", check_same_thread=False)
    provider = SingleConnectionProvider(conn)
    _seed_data(provider)
    container = AnalysisContainer(provider)
    app = create_app(container, _LEAGUE)
    return TestClient(app)


@pytest.fixture
def league() -> LeagueSettings:
    return _LEAGUE
