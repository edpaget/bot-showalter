from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.player_profile import PlayerProfileService
from tests.fakes.repos import FakePlayerRepo


def _player(
    player_id: int,
    name_first: str = "Mike",
    name_last: str = "Trout",
    bats: str | None = "R",
    throws: str | None = "R",
    birth_date: str | None = "1991-08-07",
) -> Player:
    return Player(
        id=player_id,
        name_first=name_first,
        name_last=name_last,
        bats=bats,
        throws=throws,
        birth_date=birth_date,
    )


def _valuation(player_id: int, value: float = 10.0) -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2026,
        system="zar",
        version="1.0",
        projection_system="steamer",
        projection_version="2026.1",
        player_type=PlayerType.BATTER,
        position="OF",
        value=value,
        rank=1,
        category_scores={},
    )


class TestGetProfile:
    def test_existing_player(self) -> None:
        player = _player(1)
        service = PlayerProfileService(FakePlayerRepo([player]))

        profile = service.get_profile(1, 2026)

        assert profile is not None
        assert profile.player_id == 1
        assert profile.name == "Mike Trout"
        assert profile.bats == "R"
        assert profile.throws == "R"

    def test_missing_player_returns_none(self) -> None:
        service = PlayerProfileService(FakePlayerRepo([]))

        assert service.get_profile(999, 2026) is None

    def test_age_computed_from_birth_date(self) -> None:
        # Born 1991-08-07, season 2026 → 34 as of July 1 2026
        player = _player(1, birth_date="1991-08-07")
        service = PlayerProfileService(FakePlayerRepo([player]))

        profile = service.get_profile(1, 2026)

        assert profile is not None
        assert profile.age == 34

    def test_none_birth_date_gives_none_age(self) -> None:
        player = _player(1, birth_date=None)
        service = PlayerProfileService(FakePlayerRepo([player]))

        profile = service.get_profile(1, 2026)

        assert profile is not None
        assert profile.age is None


class TestGetProfiles:
    def test_batch_lookup(self) -> None:
        players = [_player(1), _player(2, name_first="Shohei", name_last="Ohtani")]
        service = PlayerProfileService(FakePlayerRepo(players))

        profiles = service.get_profiles([1, 2], 2026)

        assert len(profiles) == 2
        assert profiles[1].name == "Mike Trout"
        assert profiles[2].name == "Shohei Ohtani"

    def test_with_positions(self) -> None:
        player = _player(1)
        service = PlayerProfileService(FakePlayerRepo([player]))
        positions = {1: ("OF", "DH")}

        profiles = service.get_profiles([1], 2026, positions=positions)

        assert profiles[1].positions == ("OF", "DH")

    def test_pitcher_type_derivation_sp(self) -> None:
        player = _player(1)
        service = PlayerProfileService(FakePlayerRepo([player]))
        positions = {1: ("SP",)}

        profiles = service.get_profiles([1], 2026, positions=positions)

        assert profiles[1].pitcher_type == "SP"

    def test_pitcher_type_derivation_rp(self) -> None:
        player = _player(1)
        service = PlayerProfileService(FakePlayerRepo([player]))
        positions = {1: ("RP",)}

        profiles = service.get_profiles([1], 2026, positions=positions)

        assert profiles[1].pitcher_type == "RP"

    def test_pitcher_type_derivation_sp_rp(self) -> None:
        player = _player(1)
        service = PlayerProfileService(FakePlayerRepo([player]))
        positions = {1: ("SP", "RP")}

        profiles = service.get_profiles([1], 2026, positions=positions)

        assert profiles[1].pitcher_type == "SP/RP"

    def test_pitcher_type_none_for_batters(self) -> None:
        player = _player(1)
        service = PlayerProfileService(FakePlayerRepo([player]))
        positions = {1: ("OF", "1B")}

        profiles = service.get_profiles([1], 2026, positions=positions)

        assert profiles[1].pitcher_type is None

    def test_missing_player_skipped(self) -> None:
        player = _player(1)
        service = PlayerProfileService(FakePlayerRepo([player]))

        profiles = service.get_profiles([1, 999], 2026)

        assert 1 in profiles
        assert 999 not in profiles


class TestEnrichValuations:
    def test_extracts_ids_from_valuations(self) -> None:
        players = [_player(1), _player(2, name_first="Shohei", name_last="Ohtani")]
        service = PlayerProfileService(FakePlayerRepo(players))
        valuations = [_valuation(1), _valuation(2)]

        profiles = service.enrich_valuations(valuations, 2026)

        assert len(profiles) == 2
        assert profiles[1].name == "Mike Trout"
        assert profiles[2].name == "Shohei Ohtani"

    def test_empty_valuations(self) -> None:
        service = PlayerProfileService(FakePlayerRepo([]))

        profiles = service.enrich_valuations([], 2026)

        assert profiles == {}
