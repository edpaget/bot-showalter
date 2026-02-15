import pytest

from fantasy_baseball_manager.domain.league_environment import LeagueEnvironment


class TestLeagueEnvironment:
    def test_construct_with_required_fields(self) -> None:
        env = LeagueEnvironment(
            league="International League",
            season=2024,
            level="AAA",
            runs_per_game=4.8,
            avg=0.260,
            obp=0.330,
            slg=0.420,
            k_pct=0.230,
            bb_pct=0.085,
            hr_per_pa=0.030,
            babip=0.300,
        )
        assert env.league == "International League"
        assert env.season == 2024
        assert env.level == "AAA"
        assert env.runs_per_game == 4.8
        assert env.avg == 0.260
        assert env.obp == 0.330
        assert env.slg == 0.420
        assert env.k_pct == 0.230
        assert env.bb_pct == 0.085
        assert env.hr_per_pa == 0.030
        assert env.babip == 0.300

    def test_optional_fields_default_to_none(self) -> None:
        env = LeagueEnvironment(
            league="International League",
            season=2024,
            level="AAA",
            runs_per_game=4.8,
            avg=0.260,
            obp=0.330,
            slg=0.420,
            k_pct=0.230,
            bb_pct=0.085,
            hr_per_pa=0.030,
            babip=0.300,
        )
        assert env.id is None
        assert env.loaded_at is None

    def test_frozen(self) -> None:
        env = LeagueEnvironment(
            league="International League",
            season=2024,
            level="AAA",
            runs_per_game=4.8,
            avg=0.260,
            obp=0.330,
            slg=0.420,
            k_pct=0.230,
            bb_pct=0.085,
            hr_per_pa=0.030,
            babip=0.300,
        )
        with pytest.raises(AttributeError):
            env.league = "Pacific Coast League"  # type: ignore[misc]
