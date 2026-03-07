from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.yahoo_draft_pick import YahooDraftPick
from fantasy_baseball_manager.domain.yahoo_league import YahooTeam
from fantasy_baseball_manager.services.draft_state import DraftConfig, DraftEngine, DraftError, DraftFormat
from fantasy_baseball_manager.services.draft_translation import (
    build_player_id_aliases,
    build_team_map,
    ingest_yahoo_pick,
    resolve_draft_position,
)


def _make_yahoo_pick(**overrides: object) -> YahooDraftPick:
    defaults: dict[str, object] = {
        "league_key": "449.l.12345",
        "season": 2026,
        "round": 1,
        "pick": 1,
        "team_key": "449.l.12345.t.1",
        "yahoo_player_key": "449.p.1234",
        "player_id": 100,
        "player_name": "Mike Trout",
        "position": "OF",
    }
    defaults.update(overrides)
    return YahooDraftPick(**defaults)  # type: ignore[arg-type]


def _make_player(player_id: int, name: str, position: str, value: float = 10.0) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=1,
        player_type="batter",
        position=position,
        value=value,
        category_z_scores={},
    )


_SNAKE_CONFIG = DraftConfig(
    teams=2,
    roster_slots={"OF": 3, "SP": 2, "C": 1, "UTIL": 1},
    format=DraftFormat.SNAKE,
    user_team=1,
    season=2026,
)

_AUCTION_CONFIG = DraftConfig(
    teams=2,
    roster_slots={"OF": 3, "SP": 2, "C": 1, "UTIL": 1},
    format=DraftFormat.AUCTION,
    user_team=1,
    season=2026,
    budget=260,
)

_TEAM_MAP = {
    "449.l.12345.t.1": 1,
    "449.l.12345.t.2": 2,
}


class TestBuildTeamMap:
    def test_maps_team_keys_to_ids(self) -> None:
        teams = [
            YahooTeam(
                team_key="449.l.12345.t.1",
                league_key="449.l.12345",
                team_id=1,
                name="Team A",
                manager_name="Alice",
                is_owned_by_user=True,
            ),
            YahooTeam(
                team_key="449.l.12345.t.2",
                league_key="449.l.12345",
                team_id=2,
                name="Team B",
                manager_name="Bob",
                is_owned_by_user=False,
            ),
        ]
        result = build_team_map(teams)
        assert result == {"449.l.12345.t.1": 1, "449.l.12345.t.2": 2}


class TestIngestYahooPick:
    def test_snake_pick_translation(self) -> None:
        players = [_make_player(100, "Mike Trout", "OF")]
        engine = DraftEngine()
        engine.start(players, _SNAKE_CONFIG)

        yahoo_pick = _make_yahoo_pick(team_key="449.l.12345.t.1")
        result = ingest_yahoo_pick(engine.pick, set(engine.state.available_pool), yahoo_pick, _TEAM_MAP)

        assert result is not None
        assert result.player_id == 100
        assert result.team == 1
        assert result.position == "OF"

    def test_auction_pick_with_cost(self) -> None:
        players = [_make_player(100, "Mike Trout", "OF")]
        engine = DraftEngine()
        engine.start(players, _AUCTION_CONFIG)

        yahoo_pick = _make_yahoo_pick(team_key="449.l.12345.t.1", cost=55)
        result = ingest_yahoo_pick(engine.pick, set(engine.state.available_pool), yahoo_pick, _TEAM_MAP)

        assert result is not None
        assert result.price == 55
        assert result.player_id == 100

    def test_unmapped_player_skipped(self) -> None:
        yahoo_pick = _make_yahoo_pick(player_id=None)
        result = ingest_yahoo_pick(lambda *a, **kw: None, frozenset(), yahoo_pick, _TEAM_MAP)

        assert result is None

    def test_player_not_in_pool_skipped(self) -> None:
        available = frozenset({200})  # player_id 100 is not in the pool
        yahoo_pick = _make_yahoo_pick(player_id=100)
        result = ingest_yahoo_pick(lambda *a, **kw: None, available, yahoo_pick, _TEAM_MAP)

        assert result is None

    def test_unknown_team_key_skipped(self) -> None:
        available = frozenset({100})
        yahoo_pick = _make_yahoo_pick(team_key="449.l.99999.t.99")
        result = ingest_yahoo_pick(lambda *a, **kw: None, available, yahoo_pick, _TEAM_MAP)

        assert result is None

    def test_position_passed_through(self) -> None:
        """ingest_yahoo_pick passes position as-is (already normalized at Yahoo boundary)."""
        players = [_make_player(100, "Mike Trout", "OF")]
        engine = DraftEngine()
        engine.start(players, _SNAKE_CONFIG)

        yahoo_pick = _make_yahoo_pick(team_key="449.l.12345.t.1", position="OF")
        result = ingest_yahoo_pick(engine.pick, set(engine.state.available_pool), yahoo_pick, _TEAM_MAP)

        assert result is not None
        assert result.position == "OF"

    def test_pick_fn_raises_draft_error_returns_none(self) -> None:
        available = frozenset({100})
        yahoo_pick = _make_yahoo_pick(team_key="449.l.12345.t.1")

        def raising_pick_fn(*args: object, **kwargs: object) -> None:
            raise DraftError("duplicate pick")

        result = ingest_yahoo_pick(raising_pick_fn, available, yahoo_pick, _TEAM_MAP)  # type: ignore[arg-type]

        assert result is None

    def test_sp_resolved_to_p_with_roster_slots(self) -> None:
        """SP position maps to P slot when roster_slots are provided."""
        config = DraftConfig(
            teams=2,
            roster_slots={"OF": 3, "P": 8, "C": 1, "BN": 4},
            format=DraftFormat.LIVE,
            user_team=1,
            season=2026,
        )
        players = [_make_player(100, "Gerrit Cole", "SP")]
        engine = DraftEngine()
        engine.start(players, config)

        yahoo_pick = _make_yahoo_pick(team_key="449.l.12345.t.1", position="SP")
        result = ingest_yahoo_pick(
            engine.pick,
            set(engine.state.available_pool),
            yahoo_pick,
            _TEAM_MAP,
            roster_slots=config.roster_slots,
            team_rosters=engine.state.team_rosters,
        )

        assert result is not None
        assert result.position == "P"

    def test_batter_overflow_to_util(self) -> None:
        """Batter position overflows to UTIL when primary slot is full."""
        config = DraftConfig(
            teams=2,
            roster_slots={"SS": 1, "UTIL": 1, "P": 2, "BN": 2},
            format=DraftFormat.LIVE,
            user_team=1,
            season=2026,
        )
        players = [
            _make_player(100, "Player A", "SS"),
            _make_player(101, "Player B", "SS"),
        ]
        engine = DraftEngine()
        engine.start(players, config)

        # First SS pick fills the SS slot
        pick1 = _make_yahoo_pick(player_id=100, player_name="Player A", team_key="449.l.12345.t.1", position="SS")
        r1 = ingest_yahoo_pick(
            engine.pick,
            set(engine.state.available_pool),
            pick1,
            _TEAM_MAP,
            roster_slots=config.roster_slots,
            team_rosters=engine.state.team_rosters,
        )
        assert r1 is not None
        assert r1.position == "SS"

        # Second SS pick overflows to UTIL
        pick2 = _make_yahoo_pick(player_id=101, player_name="Player B", team_key="449.l.12345.t.1", position="SS")
        r2 = ingest_yahoo_pick(
            engine.pick,
            set(engine.state.available_pool),
            pick2,
            _TEAM_MAP,
            roster_slots=config.roster_slots,
            team_rosters=engine.state.team_rosters,
        )
        assert r2 is not None
        assert r2.position == "UTIL"

    def test_batter_overflow_to_bn(self) -> None:
        """Batter overflows to BN when both primary and UTIL are full."""
        config = DraftConfig(
            teams=2,
            roster_slots={"SS": 1, "UTIL": 1, "P": 2, "BN": 2},
            format=DraftFormat.LIVE,
            user_team=1,
            season=2026,
        )
        players = [
            _make_player(100, "Player A", "SS"),
            _make_player(101, "Player B", "SS"),
            _make_player(102, "Player C", "SS"),
        ]
        engine = DraftEngine()
        engine.start(players, config)

        # Fill SS and UTIL
        for pid, name in [(100, "Player A"), (101, "Player B")]:
            pick = _make_yahoo_pick(player_id=pid, player_name=name, team_key="449.l.12345.t.1", position="SS")
            ingest_yahoo_pick(
                engine.pick,
                set(engine.state.available_pool),
                pick,
                _TEAM_MAP,
                roster_slots=config.roster_slots,
                team_rosters=engine.state.team_rosters,
            )

        # Third SS pick overflows to BN
        pick3 = _make_yahoo_pick(player_id=102, player_name="Player C", team_key="449.l.12345.t.1", position="SS")
        r3 = ingest_yahoo_pick(
            engine.pick,
            set(engine.state.available_pool),
            pick3,
            _TEAM_MAP,
            roster_slots=config.roster_slots,
            team_rosters=engine.state.team_rosters,
        )
        assert r3 is not None
        assert r3.position == "BN"

    def test_overflow_to_any_open_slot(self) -> None:
        """25 picks into 25 roster slots — SS unused, last OF pick lands in SS via any-open-slot."""
        config = DraftConfig(
            teams=2,
            roster_slots={"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "UTIL": 1, "P": 8, "BN": 8},
            format=DraftFormat.LIVE,
            user_team=1,
            season=2026,
        )
        # 25 players: 4 OF, 13 pitchers (SP), 7 batters at various positions, 1 more OF at end
        players = []
        pid = 1
        # 3 OF to fill OF slots
        for i in range(3):
            players.append(_make_player(pid, f"OF Player {i}", "OF"))
            pid += 1
        # 1B, 2B, 3B, C to fill those slots
        for pos in ["1B", "2B", "3B", "C"]:
            players.append(_make_player(pid, f"{pos} Player", pos))
            pid += 1
        # 1 UTIL-bound batter
        players.append(_make_player(pid, "UTIL Batter", "OF"))
        pid += 1
        # 8 pitchers to fill P slots
        for i in range(8):
            players.append(_make_player(pid, f"SP Player {i}", "SP"))
            pid += 1
        # 8 BN fillers (batters)
        for i in range(8):
            players.append(_make_player(pid, f"BN Batter {i}", "OF"))
            pid += 1
        # The 25th pick: OF that should overflow to SS
        players.append(_make_player(pid, "Wilyer Abreu", "OF"))
        last_pid = pid

        engine = DraftEngine()
        engine.start(players, config)

        # Ingest first 24 picks for team 2
        for p in players[:-1]:
            yahoo_pick = _make_yahoo_pick(
                player_id=p.player_id,
                player_name=p.player_name,
                team_key="449.l.12345.t.2",
                position=p.position,
            )
            result = ingest_yahoo_pick(
                engine.pick,
                set(engine.state.available_pool),
                yahoo_pick,
                _TEAM_MAP,
                roster_slots=config.roster_slots,
                team_rosters=engine.state.team_rosters,
            )
            assert result is not None, f"Pick {p.player_name} failed"

        # 25th pick — should land in SS (the only open slot)
        yahoo_pick_25 = _make_yahoo_pick(
            player_id=last_pid,
            player_name="Wilyer Abreu",
            team_key="449.l.12345.t.2",
            position="OF",
        )
        result_25 = ingest_yahoo_pick(
            engine.pick,
            set(engine.state.available_pool),
            yahoo_pick_25,
            _TEAM_MAP,
            roster_slots=config.roster_slots,
            team_rosters=engine.state.team_rosters,
        )
        assert result_25 is not None
        assert result_25.position == "SS"

    def test_id_alias_resolves_mismatched_id(self) -> None:
        """Yahoo ID 9007 aliased to board ID 22036 — pick uses board ID."""
        players = [_make_player(22036, "Jose Ramirez", "OF")]
        engine = DraftEngine()
        engine.start(players, _SNAKE_CONFIG)

        yahoo_pick = _make_yahoo_pick(player_id=9007, player_name="José Ramírez", position="OF")
        aliases = {9007: 22036}
        result = ingest_yahoo_pick(
            engine.pick, set(engine.state.available_pool), yahoo_pick, _TEAM_MAP, id_aliases=aliases
        )

        assert result is not None
        assert result.player_id == 22036

    def test_id_alias_not_used_when_id_already_in_pool(self) -> None:
        """Alias skipped if direct ID works."""
        players = [_make_player(100, "Mike Trout", "OF"), _make_player(200, "Other Trout", "OF")]
        engine = DraftEngine()
        engine.start(players, _SNAKE_CONFIG)

        yahoo_pick = _make_yahoo_pick(player_id=100, player_name="Mike Trout", position="OF")
        aliases = {100: 200}  # alias exists but shouldn't be used
        result = ingest_yahoo_pick(
            engine.pick, set(engine.state.available_pool), yahoo_pick, _TEAM_MAP, id_aliases=aliases
        )

        assert result is not None
        assert result.player_id == 100


class TestResolveDraftPosition:
    def test_direct_slot_with_room(self) -> None:
        assert resolve_draft_position("OF", {"OF": 3, "P": 2}, {}) == "OF"

    def test_sp_maps_to_p(self) -> None:
        assert resolve_draft_position("SP", {"P": 8, "OF": 3}, {}) == "P"

    def test_rp_maps_to_p(self) -> None:
        assert resolve_draft_position("RP", {"P": 8, "OF": 3}, {}) == "P"

    def test_sp_maps_to_bn_when_p_full(self) -> None:
        assert resolve_draft_position("SP", {"P": 1, "BN": 4}, {"P": 1}) == "BN"

    def test_batter_overflow_to_util(self) -> None:
        assert resolve_draft_position("SS", {"SS": 1, "UTIL": 1}, {"SS": 1}) == "UTIL"

    def test_batter_overflow_to_bn(self) -> None:
        assert resolve_draft_position("SS", {"SS": 1, "UTIL": 1, "BN": 4}, {"SS": 1, "UTIL": 1}) == "BN"

    def test_returns_original_when_no_fallback(self) -> None:
        assert resolve_draft_position("SS", {"SS": 1}, {"SS": 1}) == "SS"

    def test_sp_returns_original_when_no_fallback(self) -> None:
        assert resolve_draft_position("SP", {"OF": 3}, {}) == "SP"

    def test_direct_slot_partially_filled(self) -> None:
        assert resolve_draft_position("OF", {"OF": 3, "UTIL": 1}, {"OF": 1}) == "OF"

    def test_any_open_slot_fallback(self) -> None:
        """When SS, UTIL, and BN are full but OF has room, falls back to OF."""
        slots = {"SS": 1, "UTIL": 1, "BN": 1, "OF": 3}
        fills = {"SS": 1, "UTIL": 1, "BN": 1, "OF": 1}
        assert resolve_draft_position("SS", slots, fills) == "OF"

    def test_any_open_slot_when_all_full_returns_original(self) -> None:
        """All slots full → returns original position unchanged."""
        slots = {"SS": 1, "UTIL": 1, "BN": 1}
        fills = {"SS": 1, "UTIL": 1, "BN": 1}
        assert resolve_draft_position("SS", slots, fills) == "SS"


class TestBuildPlayerIdAliases:
    def test_exact_name_match_creates_alias(self) -> None:
        yahoo_picks = [_make_yahoo_pick(player_id=9007, player_name="Jose Ramirez")]
        board_names = {22036: "Jose Ramirez"}

        aliases = build_player_id_aliases(yahoo_picks, board_names)

        assert aliases == {9007: 22036}

    def test_nickname_match_creates_alias(self) -> None:
        """'Michael Busch' matches 'Mike Busch' via nickname normalization."""
        yahoo_picks = [_make_yahoo_pick(player_id=18185, player_name="Michael Busch")]
        board_names = {19681: "Mike Busch"}

        aliases = build_player_id_aliases(yahoo_picks, board_names)

        assert aliases == {18185: 19681}

    def test_accent_match_creates_alias(self) -> None:
        """'José Ramírez' matches 'Jose Ramirez' via accent stripping."""
        yahoo_picks = [_make_yahoo_pick(player_id=9007, player_name="José Ramírez")]
        board_names = {22036: "Jose Ramirez"}

        aliases = build_player_id_aliases(yahoo_picks, board_names)

        assert aliases == {9007: 22036}

    def test_no_match_no_alias(self) -> None:
        yahoo_picks = [_make_yahoo_pick(player_id=9999, player_name="Unknown Player")]
        board_names = {1: "Someone Else"}

        aliases = build_player_id_aliases(yahoo_picks, board_names)

        assert aliases == {}

    def test_already_in_board_no_alias(self) -> None:
        """Player already in board — no alias needed."""
        yahoo_picks = [_make_yahoo_pick(player_id=100, player_name="Mike Trout")]
        board_names = {100: "Mike Trout"}

        aliases = build_player_id_aliases(yahoo_picks, board_names)

        assert aliases == {}

    def test_ambiguous_name_no_alias(self) -> None:
        """Two board players match the same name — no alias created."""
        yahoo_picks = [_make_yahoo_pick(player_id=9007, player_name="Jose Ramirez")]
        board_names = {22036: "Jose Ramirez", 33333: "Jose Ramirez"}

        aliases = build_player_id_aliases(yahoo_picks, board_names)

        assert aliases == {}

    def test_parenthetical_stripped(self) -> None:
        """'Shohei Ohtani (Pitcher)' matches 'Shohei Ohtani'."""
        yahoo_picks = [_make_yahoo_pick(player_id=5000, player_name="Shohei Ohtani (Pitcher)")]
        board_names = {3771: "Shohei Ohtani"}

        aliases = build_player_id_aliases(yahoo_picks, board_names)

        assert aliases == {5000: 3771}
