from io import StringIO

import pytest
from rich.console import Console

import fantasy_baseball_manager.cli._output._valuations as _valuations_mod
from fantasy_baseball_manager.cli._output._valuations import print_valuation_regression_check
from fantasy_baseball_manager.domain.valuation import ValuationRegressionCheck


@pytest.fixture()
def _capture_console(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    buf = StringIO()
    monkeypatch.setattr(_valuations_mod, "console", Console(file=buf, highlight=False, no_color=True))
    return buf


def test_regression_check_passed(_capture_console: StringIO) -> None:
    check = ValuationRegressionCheck(passed=True, war_passed=True, hit_rate_passed=True, explanation="All good")
    print_valuation_regression_check(check)
    assert "All good" in _capture_console.getvalue()


def test_regression_check_failed(_capture_console: StringIO) -> None:
    check = ValuationRegressionCheck(passed=False, war_passed=False, hit_rate_passed=True, explanation="WAR dropped")
    print_valuation_regression_check(check)
    assert "WAR dropped" in _capture_console.getvalue()
