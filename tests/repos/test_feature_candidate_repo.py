import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.feature_candidate import FeatureCandidate
from fantasy_baseball_manager.repos.feature_candidate_repo import (
    SqliteFeatureCandidateRepo,
)


@pytest.fixture
def repo() -> SqliteFeatureCandidateRepo:
    conn = create_connection(":memory:")
    return SqliteFeatureCandidateRepo(SingleConnectionProvider(conn))


def _make_candidate(
    name: str = "barrel_ev",
    expression: str = "AVG(launch_speed) FILTER (WHERE barrel = 1)",
    player_type: str = "batter",
    min_pa: int | None = 100,
    min_ip: float | None = None,
    created_at: str = "2026-03-02",
) -> FeatureCandidate:
    return FeatureCandidate(
        name=name,
        expression=expression,
        player_type=player_type,
        min_pa=min_pa,
        min_ip=min_ip,
        created_at=created_at,
    )


class TestSqliteFeatureCandidateRepo:
    def test_save_and_retrieve(self, repo: SqliteFeatureCandidateRepo) -> None:
        candidate = _make_candidate()
        repo.save(candidate)

        result = repo.get_by_name("barrel_ev")
        assert result is not None
        assert result.name == "barrel_ev"
        assert result.expression == "AVG(launch_speed) FILTER (WHERE barrel = 1)"
        assert result.player_type == "batter"
        assert result.min_pa == 100
        assert result.min_ip is None
        assert result.created_at == "2026-03-02"

    def test_get_by_name_not_found(self, repo: SqliteFeatureCandidateRepo) -> None:
        result = repo.get_by_name("nonexistent")
        assert result is None

    def test_list_all(self, repo: SqliteFeatureCandidateRepo) -> None:
        repo.save(_make_candidate(name="a", expression="AVG(launch_speed)"))
        repo.save(_make_candidate(name="b", expression="SUM(barrel)"))
        repo.save(_make_candidate(name="c", expression="COUNT(*)"))

        results = repo.list_all()
        assert len(results) == 3
        names = {r.name for r in results}
        assert names == {"a", "b", "c"}

    def test_update_existing(self, repo: SqliteFeatureCandidateRepo) -> None:
        repo.save(_make_candidate(name="barrel_ev", expression="AVG(launch_speed)"))
        repo.save(_make_candidate(name="barrel_ev", expression="AVG(launch_speed) FILTER (WHERE barrel = 1)"))

        result = repo.get_by_name("barrel_ev")
        assert result is not None
        assert result.expression == "AVG(launch_speed) FILTER (WHERE barrel = 1)"

    def test_delete(self, repo: SqliteFeatureCandidateRepo) -> None:
        repo.save(_make_candidate(name="to_delete"))
        assert repo.get_by_name("to_delete") is not None

        deleted = repo.delete("to_delete")
        assert deleted is True
        assert repo.get_by_name("to_delete") is None

    def test_delete_nonexistent(self, repo: SqliteFeatureCandidateRepo) -> None:
        deleted = repo.delete("nonexistent")
        assert deleted is False

    def test_list_all_empty(self, repo: SqliteFeatureCandidateRepo) -> None:
        results = repo.list_all()
        assert results == []

    def test_pitcher_with_min_ip(self, repo: SqliteFeatureCandidateRepo) -> None:
        candidate = _make_candidate(
            name="k_rate",
            player_type="pitcher",
            min_pa=None,
            min_ip=50.0,
        )
        repo.save(candidate)

        result = repo.get_by_name("k_rate")
        assert result is not None
        assert result.min_ip == 50.0
        assert result.min_pa is None
