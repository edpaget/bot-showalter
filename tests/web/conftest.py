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
    PositionAppearance,
    Projection,
    RosterStint,
    StatType,
    Team,
    Valuation,
)
from fantasy_baseball_manager.repos import (
    SqliteADPRepo,
    SqliteDraftSessionRepo,
    SqlitePlayerRepo,
    SqlitePositionAppearanceRepo,
    SqliteProjectionRepo,
    SqliteRosterStintRepo,
    SqliteTeamRepo,
    SqliteValuationRepo,
)
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
    """Seed the in-memory database with test players, valuations, projections, ADP, and bio data."""
    player_repo = SqlitePlayerRepo(provider)
    valuation_repo = SqliteValuationRepo(provider)
    projection_repo = SqliteProjectionRepo(provider)
    adp_repo = SqliteADPRepo(provider)
    team_repo = SqliteTeamRepo(provider)
    roster_stint_repo = SqliteRosterStintRepo(provider)
    position_appearance_repo = SqlitePositionAppearanceRepo(provider)

    players = [
        Player(
            name_first="Mike", name_last="Trout", id=1, mlbam_id=545361, bats="R", throws="R", birth_date="1991-08-07"
        ),
        Player(
            name_first="Shohei",
            name_last="Ohtani",
            id=2,
            mlbam_id=660271,
            bats="L",
            throws="R",
            birth_date="1994-07-05",
        ),
        Player(
            name_first="Gerrit", name_last="Cole", id=3, mlbam_id=543037, bats="R", throws="R", birth_date="1990-09-08"
        ),
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

    projections = [
        Projection(
            player_id=1,
            season=2026,
            system="steamer",
            version="2026",
            player_type="batter",
            stat_json={"pa": 600, "hr": 35, "rbi": 90},
        ),
        Projection(
            player_id=2,
            season=2026,
            system="steamer",
            version="2026",
            player_type="batter",
            stat_json={"pa": 550, "hr": 40, "rbi": 100},
        ),
        Projection(
            player_id=3,
            season=2026,
            system="steamer",
            version="2026",
            player_type="pitcher",
            stat_json={"ip": 180, "w": 14, "k": 220},
        ),
    ]
    for proj in projections:
        projection_repo.upsert(proj)

    adp_records = [
        ADP(player_id=1, season=2026, provider="fantasypros", overall_pick=5.0, rank=5, positions="OF"),
        ADP(player_id=2, season=2026, provider="fantasypros", overall_pick=3.0, rank=3, positions="OF,DH"),
        ADP(player_id=3, season=2026, provider="fantasypros", overall_pick=15.0, rank=15, positions="SP"),
    ]
    for adp in adp_records:
        adp_repo.upsert(adp)

    teams = [
        Team(abbreviation="LAA", name="Los Angeles Angels", league="AL", division="West"),
        Team(abbreviation="LAD", name="Los Angeles Dodgers", league="NL", division="West"),
        Team(abbreviation="NYY", name="New York Yankees", league="AL", division="East"),
    ]
    team_ids = {}
    for t in teams:
        team_ids[t.abbreviation] = team_repo.upsert(t)

    stints = [
        RosterStint(player_id=1, team_id=team_ids["LAA"], season=2026, start_date="2026-03-28"),
        RosterStint(player_id=2, team_id=team_ids["LAD"], season=2026, start_date="2026-03-28"),
        RosterStint(player_id=3, team_id=team_ids["NYY"], season=2026, start_date="2026-03-28"),
    ]
    for s in stints:
        roster_stint_repo.upsert(s)

    appearances = [
        PositionAppearance(player_id=1, season=2026, position="CF", games=100),
        PositionAppearance(player_id=2, season=2026, position="DH", games=120),
        PositionAppearance(player_id=3, season=2026, position="SP", games=30),
    ]
    for a in appearances:
        position_appearance_repo.upsert(a)

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
