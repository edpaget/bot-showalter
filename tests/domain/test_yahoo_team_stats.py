from fantasy_baseball_manager.domain.yahoo_team_stats import TeamSeasonStats


class TestTeamSeasonStats:
    def test_field_access(self) -> None:
        stats = TeamSeasonStats(
            team_key="458.l.135575.t.10",
            league_key="458.l.135575",
            season=2025,
            team_name="My Team",
            final_rank=1,
            stat_values={"hr": 250.0, "era": 3.45},
        )
        assert stats.team_key == "458.l.135575.t.10"
        assert stats.league_key == "458.l.135575"
        assert stats.season == 2025
        assert stats.team_name == "My Team"
        assert stats.final_rank == 1
        assert stats.stat_values == {"hr": 250.0, "era": 3.45}

    def test_frozen(self) -> None:
        stats = TeamSeasonStats(
            team_key="458.l.135575.t.10",
            league_key="458.l.135575",
            season=2025,
            team_name="My Team",
            final_rank=1,
            stat_values={"hr": 250.0},
        )
        try:
            stats.team_name = "Other"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_default_id_none(self) -> None:
        stats = TeamSeasonStats(
            team_key="458.l.135575.t.10",
            league_key="458.l.135575",
            season=2025,
            team_name="My Team",
            final_rank=1,
            stat_values={},
        )
        assert stats.id is None

    def test_id_set(self) -> None:
        stats = TeamSeasonStats(
            team_key="458.l.135575.t.10",
            league_key="458.l.135575",
            season=2025,
            team_name="My Team",
            final_rank=1,
            stat_values={},
            id=42,
        )
        assert stats.id == 42
