from fantasy_baseball_manager.domain.player_profile import PlayerProfile, compute_age


class TestComputeAge:
    def test_normal_case_born_well_before_july(self) -> None:
        # Born 1990-03-15, season 2026 → 36 as of July 1 2026
        assert compute_age("1990-03-15", 2026) == 36

    def test_born_after_july_1(self) -> None:
        # Born 1990-08-20, season 2026 → still 35 on July 1 2026
        assert compute_age("1990-08-20", 2026) == 35

    def test_born_on_july_1(self) -> None:
        # Born 1990-07-01, season 2026 → exactly 36
        assert compute_age("1990-07-01", 2026) == 36

    def test_none_birth_date_returns_none(self) -> None:
        assert compute_age(None, 2026) is None


class TestPlayerProfile:
    def test_construction_with_all_fields(self) -> None:
        profile = PlayerProfile(
            player_id=1,
            name="Mike Trout",
            age=34,
            bats="R",
            throws="R",
            positions=("OF",),
            pitcher_type="SP",
        )
        assert profile.player_id == 1
        assert profile.name == "Mike Trout"
        assert profile.age == 34
        assert profile.bats == "R"
        assert profile.throws == "R"
        assert profile.positions == ("OF",)
        assert profile.pitcher_type == "SP"

    def test_defaults(self) -> None:
        profile = PlayerProfile(
            player_id=2,
            name="Test Player",
            age=None,
            bats=None,
            throws=None,
        )
        assert profile.positions == ()
        assert profile.pitcher_type is None

    def test_frozen(self) -> None:
        profile = PlayerProfile(
            player_id=1,
            name="Test",
            age=30,
            bats="L",
            throws="R",
        )
        try:
            profile.age = 31  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass
