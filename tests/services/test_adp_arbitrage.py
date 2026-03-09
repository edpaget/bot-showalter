from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.services.adp_arbitrage import (
    build_arbitrage_report,
    detect_falling_players,
    detect_reaches,
)
from fantasy_baseball_manager.services.draft_state import DraftPick


def _row(
    player_id: int,
    name: str,
    value: float,
    adp: float | None,
    rank: int = 1,
    position: str = "OF",
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=rank,
        player_type="batter",
        position=position,
        value=value,
        category_z_scores={},
        adp_overall=adp,
    )


class TestDetectFallingPlayers:
    def test_player_past_threshold_detected(self) -> None:
        available = [_row(1, "Alice", 10.0, 40.0)]
        result = detect_falling_players(55, available)
        assert len(result) == 1
        assert result[0].player_id == 1
        assert result[0].picks_past_adp == 15.0

    def test_player_within_threshold_not_detected(self) -> None:
        available = [_row(1, "Alice", 10.0, 40.0)]
        result = detect_falling_players(45, available)
        assert len(result) == 0

    def test_higher_value_scores_higher_at_same_slip(self) -> None:
        available = [
            _row(1, "HighVal", 20.0, 30.0),
            _row(2, "LowVal", 5.0, 30.0),
        ]
        result = detect_falling_players(50, available)
        assert len(result) == 2
        assert result[0].player_id == 1
        assert result[0].arbitrage_score > result[1].arbitrage_score

    def test_players_without_adp_excluded(self) -> None:
        available = [
            _row(1, "HasADP", 10.0, 30.0),
            _row(2, "NoADP", 10.0, None),
        ]
        result = detect_falling_players(50, available)
        assert len(result) == 1
        assert result[0].player_id == 1

    def test_returns_top_n_sorted_by_score(self) -> None:
        available = [_row(i, f"Player{i}", float(i), 10.0) for i in range(1, 25)]
        result = detect_falling_players(30, available, limit=5)
        assert len(result) == 5
        scores = [r.arbitrage_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_value_rank_reflects_available_pool(self) -> None:
        available = [
            _row(1, "Best", 30.0, 10.0),
            _row(2, "Mid", 20.0, 10.0),
            _row(3, "Worst", 10.0, 10.0),
        ]
        result = detect_falling_players(25, available)
        ranks = {r.player_id: r.value_rank for r in result}
        assert ranks[1] == 1
        assert ranks[2] == 2
        assert ranks[3] == 3

    def test_custom_threshold(self) -> None:
        available = [_row(1, "Alice", 10.0, 40.0)]
        # With threshold=20, pick 55 is only 15 past ADP — not enough
        assert len(detect_falling_players(55, available, threshold=20)) == 0
        # With threshold=5, pick 55 is 15 past ADP — enough
        assert len(detect_falling_players(55, available, threshold=5)) == 1


class TestDetectReaches:
    def test_reach_detected(self) -> None:
        picks = [DraftPick(pick_number=20, team=1, player_id=1, player_name="Alice", position="OF")]
        adp_lookup = {1: 45.0}
        result = detect_reaches(picks, adp_lookup)
        assert len(result) == 1
        assert result[0].picks_ahead_of_adp == 25.0
        assert result[0].drafter_team == 1

    def test_small_reach_not_detected(self) -> None:
        picks = [DraftPick(pick_number=40, team=1, player_id=1, player_name="Alice", position="OF")]
        adp_lookup = {1: 45.0}
        result = detect_reaches(picks, adp_lookup)
        assert len(result) == 0

    def test_players_without_adp_excluded(self) -> None:
        picks = [DraftPick(pick_number=20, team=1, player_id=1, player_name="Alice", position="OF")]
        adp_lookup: dict[int, float] = {}
        result = detect_reaches(picks, adp_lookup)
        assert len(result) == 0

    def test_sorted_by_picks_ahead_descending(self) -> None:
        picks = [
            DraftPick(pick_number=10, team=1, player_id=1, player_name="A", position="OF"),
            DraftPick(pick_number=5, team=2, player_id=2, player_name="B", position="OF"),
        ]
        adp_lookup = {1: 50.0, 2: 80.0}
        result = detect_reaches(picks, adp_lookup)
        assert len(result) == 2
        assert result[0].player_id == 2  # 80 - 5 = 75 ahead
        assert result[1].player_id == 1  # 50 - 10 = 40 ahead


class TestBuildArbitrageReport:
    def test_combines_falling_and_reaches(self) -> None:
        available = [_row(1, "Faller", 10.0, 20.0)]
        picks = [DraftPick(pick_number=5, team=2, player_id=2, player_name="Reacher", position="SS")]
        adp_lookup = {2: 40.0}
        report = build_arbitrage_report(40, available, picks, adp_lookup)
        assert report.current_pick == 40
        assert len(report.falling) == 1
        assert len(report.reaches) == 1
