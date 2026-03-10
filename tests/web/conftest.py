import pytest
from fastapi.testclient import TestClient

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import (
    ADP,
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    Player,
    StatType,
    Valuation,
)
from fantasy_baseball_manager.repos import SqliteADPRepo, SqliteDraftSessionRepo, SqlitePlayerRepo, SqliteValuationRepo
from fantasy_baseball_manager.web import SessionManager, create_app

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

    adp_repo = SqliteADPRepo(provider)
    adp_records = [
        ADP(player_id=1, season=2026, provider="fantasypros", overall_pick=5.0, rank=5, positions="OF"),
        ADP(player_id=2, season=2026, provider="fantasypros", overall_pick=8.0, rank=8, positions="OF"),
        ADP(player_id=3, season=2026, provider="fantasypros", overall_pick=12.0, rank=12, positions="SP"),
    ]
    for a in adp_records:
        adp_repo.upsert(a)

    with provider.connection() as conn:
        conn.commit()


def _make_provider() -> SingleConnectionProvider:
    conn = create_connection(":memory:", check_same_thread=False)
    provider = SingleConnectionProvider(conn)
    _seed_data(provider)
    return provider


@pytest.fixture
def client() -> TestClient:
    """Create a test client with an in-memory SQLite database seeded with test data."""
    provider = _make_provider()
    container = AnalysisContainer(provider)
    app = create_app(container, _LEAGUE, default_system="zar", default_version="1.0")
    return TestClient(app)


@pytest.fixture
def session_client() -> TestClient:
    """Create a test client with SessionManager enabled."""
    provider = _make_provider()
    container = AnalysisContainer(provider)
    session_repo = SqliteDraftSessionRepo(provider)
    session_manager = SessionManager(
        session_repo=session_repo,
        valuation_repo=container.valuation_repo,
        player_repo=container.player_repo,
        adp_repo=container.adp_repo,
        player_profile_service=container.player_profile_service,
        league=_LEAGUE,
        adp_provider="fantasypros",
    )
    app = create_app(container, _LEAGUE, session_manager=session_manager, default_system="zar", default_version="1.0")
    return TestClient(app)


@pytest.fixture
def session_provider() -> SingleConnectionProvider:
    """Return a seeded provider for tests needing direct DB access."""
    return _make_provider()


@pytest.fixture
def league() -> LeagueSettings:
    return _LEAGUE
