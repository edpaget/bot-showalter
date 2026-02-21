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
from fantasy_baseball_manager.services.draft_board import build_draft_board, export_csv, export_html


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


# --- export_html tests ---


class TestExportHtmlBasic:
    def test_valid_html5_document(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Mike Trout",
                rank=1,
                player_type="batter",
                position="OF",
                value=42.5,
                category_z_scores={"hr": 2.1},
            ),
        ]
        board = _board_with_rows(rows)
        league = _league()
        buf = io.StringIO()
        export_html(board, league, buf)
        html = buf.getvalue()
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "<table" in html
        assert "</html>" in html

    def test_league_name_in_title(self) -> None:
        board = _board_with_rows(
            [
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
        )
        league = _league()
        buf = io.StringIO()
        export_html(board, league, buf)
        html = buf.getvalue()
        assert "<title>" in html
        assert "Test League" in html

    def test_player_names_in_table(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Mike Trout",
                rank=1,
                player_type="batter",
                position="OF",
                value=42.5,
                category_z_scores={},
            ),
            DraftBoardRow(
                player_id=2,
                player_name="Gerrit Cole",
                rank=2,
                player_type="pitcher",
                position="SP",
                value=35.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "Mike Trout" in html
        assert "Gerrit Cole" in html

    def test_html_escapes_player_names(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="O'Brien <script>",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "O&#x27;Brien &lt;script&gt;" in html
        assert "<script>" not in html

    def test_correct_column_headers(self) -> None:
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
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "<th>Rank</th>" in html
        assert "<th>Player</th>" in html
        assert "<th>Pos</th>" in html
        assert "<th>Value</th>" in html

    def test_conditional_tier_column(self) -> None:
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
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "<th>Tier</th>" in html

    def test_no_tier_column_when_absent(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "<th>Tier</th>" not in html

    def test_conditional_adp_columns(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
                adp_overall=15.0,
                adp_rank=15,
                adp_delta=14,
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "<th>ADP</th>" in html
        assert "<th>ADPRk</th>" in html
        assert "<th>Delta</th>" in html

    def test_no_adp_columns_when_absent(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "<th>ADP</th>" not in html
        assert "<th>Delta</th>" not in html


class TestExportHtmlTierColors:
    def test_tier_rows_get_css_class(self) -> None:
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
                tier=3,
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert 'class="tier-1"' in html
        assert 'class="tier-3"' in html

    def test_css_defines_tier_colors(self) -> None:
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
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert ".tier-1" in html
        assert ".tier-2" in html
        assert "background" in html

    def test_no_tier_class_when_tier_is_none(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
                tier=None,
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        # The player row should not have a tier class
        assert 'class="tier-' not in html

    def test_tier_wraps_with_mod_8(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
                tier=9,
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        # tier 9 mod 8 = 1
        assert 'class="tier-1"' in html


class TestExportHtmlAdpHighlight:
    def test_buy_class_on_positive_delta(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
                adp_overall=15.0,
                adp_rank=15,
                adp_delta=15,
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf, adp_delta_threshold=10)
        html = buf.getvalue()
        assert 'class="buy"' in html

    def test_avoid_class_on_negative_delta(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
                adp_overall=15.0,
                adp_rank=15,
                adp_delta=-15,
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf, adp_delta_threshold=10)
        html = buf.getvalue()
        assert 'class="avoid"' in html

    def test_no_highlight_within_threshold(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
                adp_overall=15.0,
                adp_rank=15,
                adp_delta=5,
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf, adp_delta_threshold=10)
        html = buf.getvalue()
        assert 'class="buy"' not in html
        assert 'class="avoid"' not in html


class TestExportHtmlGrouping:
    def test_batters_before_pitchers(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Pitcher A",
                rank=1,
                player_type="pitcher",
                position="SP",
                value=50.0,
                category_z_scores={},
            ),
            DraftBoardRow(
                player_id=2,
                player_name="Batter A",
                rank=2,
                player_type="batter",
                position="OF",
                value=40.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        batter_pos = html.index("Batter A")
        pitcher_pos = html.index("Pitcher A")
        assert batter_pos < pitcher_pos

    def test_section_headers_present(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Batter A",
                rank=1,
                player_type="batter",
                position="OF",
                value=50.0,
                category_z_scores={},
            ),
            DraftBoardRow(
                player_id=2,
                player_name="Pitcher A",
                rank=2,
                player_type="pitcher",
                position="SP",
                value=40.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "Batters" in html
        assert "Pitchers" in html

    def test_position_subgroup_headers(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="A",
                rank=1,
                player_type="batter",
                position="C",
                value=20.0,
                category_z_scores={},
            ),
            DraftBoardRow(
                player_id=2,
                player_name="B",
                rank=2,
                player_type="batter",
                position="OF",
                value=15.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert ">C<" in html
        assert ">OF<" in html

    def test_position_order_batters(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="OF guy",
                rank=1,
                player_type="batter",
                position="OF",
                value=50.0,
                category_z_scores={},
            ),
            DraftBoardRow(
                player_id=2,
                player_name="C guy",
                rank=2,
                player_type="batter",
                position="C",
                value=40.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        c_pos = html.index("C guy")
        of_pos = html.index("OF guy")
        assert c_pos < of_pos

    def test_within_position_sorted_by_value_desc(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Low OF",
                rank=2,
                player_type="batter",
                position="OF",
                value=10.0,
                category_z_scores={},
            ),
            DraftBoardRow(
                player_id=2,
                player_name="High OF",
                rank=1,
                player_type="batter",
                position="OF",
                value=30.0,
                category_z_scores={},
            ),
        ]
        board = _board_with_rows(rows)
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        high_pos = html.index("High OF")
        low_pos = html.index("Low OF")
        assert high_pos < low_pos


class TestExportHtmlPrintCss:
    def test_contains_media_print(self) -> None:
        board = _board_with_rows(
            [
                DraftBoardRow(
                    player_id=1,
                    player_name="A",
                    rank=1,
                    player_type="batter",
                    position="OF",
                    value=10.0,
                    category_z_scores={},
                ),
            ]
        )
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "@media print" in html

    def test_landscape_page(self) -> None:
        board = _board_with_rows(
            [
                DraftBoardRow(
                    player_id=1,
                    player_name="A",
                    rank=1,
                    player_type="batter",
                    position="OF",
                    value=10.0,
                    category_z_scores={},
                ),
            ]
        )
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "@page" in html
        assert "landscape" in html

    def test_no_external_resources(self) -> None:
        board = _board_with_rows(
            [
                DraftBoardRow(
                    player_id=1,
                    player_name="A",
                    rank=1,
                    player_type="batter",
                    position="OF",
                    value=10.0,
                    category_z_scores={},
                ),
            ]
        )
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "@font-face" not in html
        assert "@import" not in html
        assert "url(" not in html

    def test_page_break_avoid_on_tr(self) -> None:
        board = _board_with_rows(
            [
                DraftBoardRow(
                    player_id=1,
                    player_name="A",
                    rank=1,
                    player_type="batter",
                    position="OF",
                    value=10.0,
                    category_z_scores={},
                ),
            ]
        )
        buf = io.StringIO()
        export_html(board, _league(), buf)
        html = buf.getvalue()
        assert "page-break-inside: avoid" in html


class TestExportHtmlSmoke:
    def test_realistic_mixed_board(self) -> None:
        rows = [
            DraftBoardRow(
                player_id=1,
                player_name="Mike Trout",
                rank=1,
                player_type="batter",
                position="OF",
                value=42.5,
                category_z_scores={"hr": 2.1, "r": 1.0, "rbi": 0.8},
                tier=1,
                adp_overall=3.0,
                adp_rank=3,
                adp_delta=2,
            ),
            DraftBoardRow(
                player_id=2,
                player_name="Aaron Judge",
                rank=2,
                player_type="batter",
                position="OF",
                value=40.0,
                category_z_scores={"hr": 2.5, "r": 0.9, "rbi": 1.1},
                tier=1,
                adp_overall=1.0,
                adp_rank=1,
                adp_delta=-1,
            ),
            DraftBoardRow(
                player_id=3,
                player_name="Trea Turner",
                rank=3,
                player_type="batter",
                position="SS",
                value=35.0,
                category_z_scores={"hr": 0.5, "r": 1.8, "rbi": 0.3},
                tier=2,
                adp_overall=10.0,
                adp_rank=10,
                adp_delta=7,
            ),
            DraftBoardRow(
                player_id=4,
                player_name="Adley Rutschman",
                rank=4,
                player_type="batter",
                position="C",
                value=28.0,
                category_z_scores={"hr": 0.3, "r": 0.5, "rbi": 0.4},
                tier=2,
                adp_overall=25.0,
                adp_rank=25,
                adp_delta=21,
            ),
            DraftBoardRow(
                player_id=5,
                player_name="Jose Ramirez",
                rank=5,
                player_type="batter",
                position="3B",
                value=38.0,
                category_z_scores={"hr": 1.0, "r": 1.2, "rbi": 1.5},
                tier=1,
                adp_overall=5.0,
                adp_rank=5,
                adp_delta=0,
            ),
            DraftBoardRow(
                player_id=6,
                player_name="Gerrit Cole",
                rank=6,
                player_type="pitcher",
                position="SP",
                value=30.0,
                category_z_scores={"w": 1.5, "sv": -0.2, "era": 1.8},
                tier=3,
                adp_overall=8.0,
                adp_rank=8,
                adp_delta=2,
            ),
            DraftBoardRow(
                player_id=7,
                player_name="Josh Hader",
                rank=7,
                player_type="pitcher",
                position="RP",
                value=20.0,
                category_z_scores={"w": -0.5, "sv": 2.0, "era": 0.5},
                tier=4,
                adp_overall=50.0,
                adp_rank=50,
                adp_delta=43,
            ),
            DraftBoardRow(
                player_id=8,
                player_name="Spencer Strider",
                rank=8,
                player_type="pitcher",
                position="SP",
                value=25.0,
                category_z_scores={"w": 1.0, "sv": 0.0, "era": 1.2},
                tier=3,
                adp_overall=12.0,
                adp_rank=12,
                adp_delta=4,
            ),
        ]
        board = _board_with_rows(rows)
        league = _league()
        buf = io.StringIO()
        export_html(board, league, buf, adp_delta_threshold=10)
        html = buf.getvalue()

        # Valid HTML5
        assert "<!DOCTYPE html>" in html
        assert "<table" in html

        # League name
        assert "Test League" in html

        # All player names present
        for row in rows:
            assert row.player_name in html

        # Batters before pitchers
        assert html.index("Mike Trout") < html.index("Gerrit Cole")

        # Section headers
        assert "Batters" in html
        assert "Pitchers" in html

        # Tier colors
        assert 'class="tier-1"' in html
        assert 'class="tier-2"' in html
        assert 'class="tier-3"' in html
        assert 'class="tier-4"' in html

        # ADP columns
        assert "<th>ADP</th>" in html
        assert "<th>Delta</th>" in html

        # Buy highlight (Adley Rutschman delta=21 >= 10)
        assert 'class="buy"' in html

        # Print CSS
        assert "@media print" in html
        assert "landscape" in html
