from io import StringIO

from rich.console import Console

from fantasy_baseball_manager.cli._output import print_talent_delta_report
from fantasy_baseball_manager.domain.performance_delta import PlayerStatDelta


def _capture_output(title: str, deltas: list[PlayerStatDelta], top: int | None = None) -> str:
    buf = StringIO()
    console = Console(file=buf, highlight=False, width=120)
    print_talent_delta_report(title, deltas, top=top, console=console)
    return buf.getvalue()


class TestPrintTalentDeltaReportGroupsByStat:
    def test_print_talent_delta_report_groups_by_stat(self) -> None:
        deltas = [
            PlayerStatDelta(
                player_id=1,
                player_name="J. Soto",
                stat_name="avg",
                actual=0.310,
                expected=0.285,
                delta=0.025,
                performance_delta=0.025,
                percentile=98.0,
            ),
            PlayerStatDelta(
                player_id=2,
                player_name="M. Trout",
                stat_name="avg",
                actual=0.220,
                expected=0.258,
                delta=-0.038,
                performance_delta=-0.038,
                percentile=3.0,
            ),
            PlayerStatDelta(
                player_id=3,
                player_name="A. Judge",
                stat_name="obp",
                actual=0.400,
                expected=0.380,
                delta=0.020,
                performance_delta=0.020,
                percentile=90.0,
            ),
            PlayerStatDelta(
                player_id=4,
                player_name="B. Harper",
                stat_name="obp",
                actual=0.320,
                expected=0.350,
                delta=-0.030,
                performance_delta=-0.030,
                percentile=10.0,
            ),
        ]

        output = _capture_output("Test Report", deltas)

        # Title is printed
        assert "Test Report" in output

        # Both stats have sections
        assert "avg" in output
        assert "obp" in output

        # Regression candidates section exists
        assert "Regression Candidates" in output

        # Buy-low section exists
        assert "Buy-Low Targets" in output

        # Players appear in the output
        assert "J. Soto" in output
        assert "M. Trout" in output
        assert "A. Judge" in output
        assert "B. Harper" in output

    def test_top_limits_per_direction(self) -> None:
        deltas = [
            PlayerStatDelta(1, "P1", "avg", 0.310, 0.280, 0.030, 0.030, 95.0),
            PlayerStatDelta(2, "P2", "avg", 0.305, 0.280, 0.025, 0.025, 90.0),
            PlayerStatDelta(3, "P3", "avg", 0.300, 0.280, 0.020, 0.020, 85.0),
            PlayerStatDelta(4, "P4", "avg", 0.250, 0.280, -0.030, -0.030, 5.0),
        ]

        output = _capture_output("Top Test", deltas, top=1)

        # Only top 1 regression candidate should appear (P1)
        assert "P1" in output
        # P2 and P3 should be excluded by top=1
        assert "P2" not in output
        assert "P3" not in output
        # Buy-low still shows P4 (only 1 in that direction)
        assert "P4" in output
