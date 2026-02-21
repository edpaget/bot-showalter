import csv
import io

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.draft_board import (
    DraftBoard,
    DraftBoardRow,
    TierAssignment,
)
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.draft_board import build_draft_board, export_csv


def _league(
    batting_keys: tuple[str, ...] = ("hr", "r", "rbi"),
    pitching_keys: tuple[str, ...] = ("w", "sv", "era"),
) -> LeagueSettings:
    batting_cats = tuple(
        CategoryConfig(key=k, name=k.upper(), stat_type=StatType.COUNTING, direction=Direction.HIGHER)
        for k in batting_keys
    )
    pitching_cats = tuple(
        CategoryConfig(key=k, name=k.upper(), stat_type=StatType.COUNTING, direction=Direction.HIGHER)
        for k in pitching_keys
    )
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=14,
        roster_pitchers=9,
        batting_categories=batting_cats,
        pitching_categories=pitching_cats,
    )


def _valuation(
    player_id: int,
    value: float,
    player_type: str = "batter",
    position: str = "OF",
    category_scores: dict[str, float] | None = None,
) -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2026,
        system="zar",
        version="1.0",
        projection_system="steamer",
        projection_version="2026.1",
        player_type=player_type,
        position=position,
        value=value,
        rank=1,
        category_scores=category_scores or {},
    )


def _names(*ids: int) -> dict[int, str]:
    return {pid: f"Player {pid}" for pid in ids}


class TestRowsRankedByValueDescending:
    def test_three_valuations_ranked_by_value(self) -> None:
        valuations = [
            _valuation(1, value=10.0),
            _valuation(2, value=30.0),
            _valuation(3, value=20.0),
        ]
        board = build_draft_board(valuations, _league(), _names(1, 2, 3))

        assert [r.player_id for r in board.rows] == [2, 3, 1]
        assert [r.rank for r in board.rows] == [1, 2, 3]

    def test_rank_is_one_indexed(self) -> None:
        valuations = [_valuation(1, value=5.0)]
        board = build_draft_board(valuations, _league(), _names(1))

        assert board.rows[0].rank == 1


class TestCategoryZScoresFiltered:
    def test_batter_gets_only_batting_categories(self) -> None:
        valuations = [
            _valuation(
                1,
                value=10.0,
                player_type="batter",
                category_scores={"hr": 1.5, "r": 0.8, "rbi": 1.2, "w": 0.0, "extra": 9.9},
            ),
        ]
        league = _league(batting_keys=("hr", "r", "rbi"), pitching_keys=("w",))
        board = build_draft_board(valuations, league, _names(1))

        row = board.rows[0]
        assert row.category_z_scores == {"hr": 1.5, "r": 0.8, "rbi": 1.2}

    def test_pitcher_gets_only_pitching_categories(self) -> None:
        valuations = [
            _valuation(
                1,
                value=15.0,
                player_type="pitcher",
                position="SP",
                category_scores={"w": 2.0, "sv": 0.5, "era": -1.0, "hr": 0.0},
            ),
        ]
        league = _league(batting_keys=("hr",), pitching_keys=("w", "sv", "era"))
        board = build_draft_board(valuations, league, _names(1))

        row = board.rows[0]
        assert row.category_z_scores == {"w": 2.0, "sv": 0.5, "era": -1.0}

    def test_missing_category_omitted_from_z_scores(self) -> None:
        valuations = [
            _valuation(1, value=10.0, category_scores={"hr": 1.0}),
        ]
        league = _league(batting_keys=("hr", "r", "rbi"))
        board = build_draft_board(valuations, league, _names(1))

        assert board.rows[0].category_z_scores == {"hr": 1.0}


class TestBoardMetadata:
    def test_batting_and_pitching_category_tuples(self) -> None:
        league = _league(batting_keys=("hr", "sb"), pitching_keys=("w", "era"))
        board = build_draft_board([], league, {})

        assert board.batting_categories == ("hr", "sb")
        assert board.pitching_categories == ("w", "era")

    def test_empty_input_returns_empty_board(self) -> None:
        board = build_draft_board([], _league(), {})

        assert board.rows == []
        assert isinstance(board, DraftBoard)


class TestTierEnrichment:
    def test_tiers_joined_by_player_id(self) -> None:
        valuations = [_valuation(1, value=10.0), _valuation(2, value=5.0)]
        tiers = [TierAssignment(player_id=1, tier=1), TierAssignment(player_id=2, tier=3)]
        board = build_draft_board(valuations, _league(), _names(1, 2), tiers=tiers)

        assert board.rows[0].tier == 1
        assert board.rows[1].tier == 3

    def test_tier_none_when_tiers_not_provided(self) -> None:
        valuations = [_valuation(1, value=10.0)]
        board = build_draft_board(valuations, _league(), _names(1))

        assert board.rows[0].tier is None

    def test_tier_none_for_unmatched_player(self) -> None:
        valuations = [_valuation(1, value=10.0), _valuation(2, value=5.0)]
        tiers = [TierAssignment(player_id=1, tier=2)]
        board = build_draft_board(valuations, _league(), _names(1, 2), tiers=tiers)

        assert board.rows[0].tier == 2
        assert board.rows[1].tier is None


class TestADPEnrichment:
    def test_adp_joined_by_player_id(self) -> None:
        valuations = [_valuation(1, value=20.0)]
        adp_list = [ADP(player_id=1, season=2026, provider="fp", overall_pick=15.0, rank=15, positions="OF")]
        board = build_draft_board(valuations, _league(), _names(1), adp=adp_list)

        row = board.rows[0]
        assert row.adp_overall == 15.0
        assert row.adp_rank == 15

    def test_adp_delta_computed(self) -> None:
        valuations = [_valuation(1, value=20.0)]
        adp_list = [ADP(player_id=1, season=2026, provider="fp", overall_pick=25.0, rank=25, positions="OF")]
        board = build_draft_board(valuations, _league(), _names(1), adp=adp_list)

        row = board.rows[0]
        assert row.rank == 1
        assert row.adp_rank == 25
        assert row.adp_delta == 24  # 25 - 1 = positive means market undervalues

    def test_adp_none_when_not_provided(self) -> None:
        valuations = [_valuation(1, value=10.0)]
        board = build_draft_board(valuations, _league(), _names(1))

        row = board.rows[0]
        assert row.adp_overall is None
        assert row.adp_rank is None
        assert row.adp_delta is None

    def test_adp_none_for_unmatched_player(self) -> None:
        valuations = [_valuation(1, value=20.0), _valuation(2, value=10.0)]
        adp_list = [ADP(player_id=1, season=2026, provider="fp", overall_pick=5.0, rank=5, positions="OF")]
        board = build_draft_board(valuations, _league(), _names(1, 2), adp=adp_list)

        assert board.rows[0].adp_rank == 5
        assert board.rows[1].adp_overall is None
        assert board.rows[1].adp_rank is None
        assert board.rows[1].adp_delta is None

    def test_duplicate_adp_uses_lowest_pick(self) -> None:
        valuations = [_valuation(1, value=20.0)]
        adp_list = [
            ADP(player_id=1, season=2026, provider="fp", overall_pick=30.0, rank=30, positions="OF"),
            ADP(player_id=1, season=2026, provider="fp", overall_pick=10.0, rank=10, positions="OF"),
        ]
        board = build_draft_board(valuations, _league(), _names(1), adp=adp_list)

        assert board.rows[0].adp_overall == 10.0
        assert board.rows[0].adp_rank == 10


class TestTwoWayADP:
    def test_pitcher_valuation_prefers_pitcher_adp(self) -> None:
        valuations = [_valuation(1, value=25.0, player_type="pitcher", position="SP")]
        adp_list = [
            ADP(player_id=1, season=2026, provider="fp", overall_pick=50.0, rank=50, positions="DH"),
            ADP(player_id=1, season=2026, provider="fp", overall_pick=20.0, rank=20, positions="SP"),
        ]
        league = _league(pitching_keys=("w", "sv", "era"))
        board = build_draft_board(valuations, league, _names(1), adp=adp_list)

        assert board.rows[0].adp_overall == 20.0
        assert board.rows[0].adp_rank == 20

    def test_batter_valuation_prefers_batter_adp(self) -> None:
        valuations = [_valuation(1, value=25.0, player_type="batter", position="DH")]
        adp_list = [
            ADP(player_id=1, season=2026, provider="fp", overall_pick=15.0, rank=15, positions="SP"),
            ADP(player_id=1, season=2026, provider="fp", overall_pick=30.0, rank=30, positions="DH"),
        ]
        board = build_draft_board(valuations, _league(), _names(1), adp=adp_list)

        assert board.rows[0].adp_overall == 30.0
        assert board.rows[0].adp_rank == 30

    def test_fallback_to_lowest_pick_when_no_position_match(self) -> None:
        valuations = [_valuation(1, value=25.0, player_type="pitcher", position="SP")]
        adp_list = [
            ADP(player_id=1, season=2026, provider="fp", overall_pick=40.0, rank=40, positions="DH"),
            ADP(player_id=1, season=2026, provider="fp", overall_pick=20.0, rank=20, positions="OF"),
        ]
        league = _league(pitching_keys=("w",))
        board = build_draft_board(valuations, league, _names(1), adp=adp_list)

        assert board.rows[0].adp_overall == 20.0


class TestMixedPool:
    def test_batters_and_pitchers_interleaved_by_value(self) -> None:
        valuations = [
            _valuation(1, value=30.0, player_type="batter", category_scores={"hr": 2.0, "r": 1.0}),
            _valuation(2, value=25.0, player_type="pitcher", position="SP", category_scores={"w": 1.5, "sv": 0.5}),
            _valuation(3, value=20.0, player_type="batter", category_scores={"hr": 1.0, "r": 0.5}),
        ]
        league = _league(batting_keys=("hr", "r"), pitching_keys=("w", "sv"))
        board = build_draft_board(valuations, league, _names(1, 2, 3))

        assert [r.player_id for r in board.rows] == [1, 2, 3]
        assert [r.rank for r in board.rows] == [1, 2, 3]
        assert board.rows[0].category_z_scores == {"hr": 2.0, "r": 1.0}
        assert board.rows[1].category_z_scores == {"w": 1.5, "sv": 0.5}
        assert board.rows[2].category_z_scores == {"hr": 1.0, "r": 0.5}


class TestReCallability:
    def test_rebuild_with_shrinking_pool(self) -> None:
        valuations = [
            _valuation(1, value=30.0),
            _valuation(2, value=20.0),
            _valuation(3, value=10.0),
        ]
        board_full = build_draft_board(valuations, _league(), _names(1, 2, 3))
        assert [r.player_id for r in board_full.rows] == [1, 2, 3]
        assert [r.rank for r in board_full.rows] == [1, 2, 3]

        remaining = [v for v in valuations if v.player_id != 1]
        board_after = build_draft_board(remaining, _league(), _names(2, 3))
        assert [r.player_id for r in board_after.rows] == [2, 3]
        assert [r.rank for r in board_after.rows] == [1, 2]


class TestPlayerNames:
    def test_names_from_map(self) -> None:
        valuations = [_valuation(1, value=10.0)]
        names = {1: "Mike Trout"}
        board = build_draft_board(valuations, _league(), names)

        assert board.rows[0].player_name == "Mike Trout"

    def test_unknown_id_gets_fallback(self) -> None:
        valuations = [_valuation(99, value=10.0)]
        board = build_draft_board(valuations, _league(), {})

        assert board.rows[0].player_name == "Unknown (99)"


class TestRowFields:
    def test_row_carries_position_and_player_type(self) -> None:
        valuations = [_valuation(1, value=10.0, player_type="batter", position="SS")]
        board = build_draft_board(valuations, _league(), _names(1))

        row = board.rows[0]
        assert row.player_type == "batter"
        assert row.position == "SS"
        assert row.value == 10.0
        assert row.player_id == 1

    def test_row_is_frozen(self) -> None:
        row = DraftBoardRow(
            player_id=1,
            player_name="Test",
            rank=1,
            player_type="batter",
            position="OF",
            value=10.0,
            category_z_scores={},
        )
        assert row.rank == 1


def _board_with_rows(
    rows: list[DraftBoardRow],
    batting_cats: tuple[str, ...] = ("hr", "r", "rbi"),
    pitching_cats: tuple[str, ...] = ("w", "sv", "era"),
) -> DraftBoard:
    return DraftBoard(rows=rows, batting_categories=batting_cats, pitching_categories=pitching_cats)


class TestExportCsvBasic:
    def test_basic_batter_and_pitcher(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Mike Trout",
                rank=1,
                player_type="batter",
                position="OF",
                value=42.5,
                category_z_scores={"hr": 2.1, "r": 1.0, "rbi": 0.8},
            ),
            DraftBoardRow(
                player_id=2,
                player_name="Gerrit Cole",
                rank=2,
                player_type="pitcher",
                position="SP",
                value=35.0,
                category_z_scores={"w": 1.5, "sv": -0.2, "era": 1.8},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_csv(board, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        csv_rows = list(reader)

        assert len(csv_rows) == 2
        assert csv_rows[0]["Player"] == "Mike Trout"
        assert csv_rows[0]["Rank"] == "1"
        assert csv_rows[0]["Value"] == "$42.5"
        assert csv_rows[1]["Player"] == "Gerrit Cole"

    def test_header_contains_expected_columns(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Test",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={"hr": 1.0},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_csv(board, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        fieldnames = reader.fieldnames or []
        assert "Rank" in fieldnames
        assert "Player" in fieldnames
        assert "Type" in fieldnames
        assert "Pos" in fieldnames
        assert "Value" in fieldnames
        assert "hr" in fieldnames
        assert "w" in fieldnames


class TestExportCsvOmitColumns:
    def test_omits_tier_and_adp_when_all_none(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Test",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_csv(board, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        fieldnames = reader.fieldnames or []
        assert "Tier" not in fieldnames
        assert "ADP" not in fieldnames
        assert "ADPRk" not in fieldnames
        assert "Delta" not in fieldnames

    def test_includes_tier_when_any_present(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
                tier=1,
            ),
            DraftBoardRow(
                player_id=2,
                player_name="B",
                rank=2,
                player_type="batter",
                position="1B",
                value=5.0,
                category_z_scores={},
                tier=None,
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_csv(board, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        fieldnames = reader.fieldnames or []
        assert "Tier" in fieldnames
        csv_rows = list(reader)
        assert csv_rows[0]["Tier"] == "1"
        assert csv_rows[1]["Tier"] == ""

    def test_includes_adp_when_any_present(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
                adp_overall=15.5,
                adp_rank=15,
                adp_delta=14,
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_csv(board, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        fieldnames = reader.fieldnames or []
        assert "ADP" in fieldnames
        assert "ADPRk" in fieldnames
        assert "Delta" in fieldnames
        csv_rows = list(reader)
        assert csv_rows[0]["ADP"] == "15.5"
        assert csv_rows[0]["ADPRk"] == "15"
        assert csv_rows[0]["Delta"] == "14"


class TestExportCsvEmpty:
    def test_empty_board_produces_header_only(self) -> None:
        board = _board_with_rows([])
        buf = io.StringIO()
        export_csv(board, buf)
        buf.seek(0)
        content = buf.getvalue()
        lines = content.strip().split("\n")
        assert len(lines) == 1  # header only
        assert "Rank" in lines[0]


class TestExportCsvZScoreFormatting:
    def test_batter_has_batting_z_scores_and_empty_pitching(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Batter",
                rank=1,
                player_type="batter",
                position="OF",
                value=20.0,
                category_z_scores={"hr": 1.23, "r": 0.45, "rbi": -0.67},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_csv(board, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        csv_rows = list(reader)
        row = csv_rows[0]
        assert row["hr"] == "1.23"
        assert row["r"] == "0.45"
        assert row["rbi"] == "-0.67"
        # Pitching columns should be empty for a batter
        assert row["w"] == ""
        assert row["sv"] == ""
        assert row["era"] == ""

    def test_pitcher_has_pitching_z_scores_and_empty_batting(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Pitcher",
                rank=1,
                player_type="pitcher",
                position="SP",
                value=15.0,
                category_z_scores={"w": 1.50, "sv": 0.00, "era": -1.20},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_csv(board, buf)
        buf.seek(0)
        reader = csv.DictReader(buf)
        csv_rows = list(reader)
        row = csv_rows[0]
        # Batting columns should be empty for a pitcher
        assert row["hr"] == ""
        assert row["r"] == ""
        assert row["rbi"] == ""
        assert row["w"] == "1.50"
        assert row["sv"] == "0.00"
        assert row["era"] == "-1.20"
