import datetime

from fantasy_baseball_manager.domain.season import current_season


class TestCurrentSeason:
    def test_mid_season_returns_current_year(self) -> None:
        assert current_season(datetime.date(2025, 6, 15)) == 2025

    def test_january_returns_current_year(self) -> None:
        assert current_season(datetime.date(2026, 1, 1)) == 2026

    def test_september_returns_current_year(self) -> None:
        assert current_season(datetime.date(2025, 9, 30)) == 2025

    def test_october_returns_next_year(self) -> None:
        assert current_season(datetime.date(2025, 10, 1)) == 2026

    def test_december_returns_next_year(self) -> None:
        assert current_season(datetime.date(2025, 12, 31)) == 2026

    def test_defaults_to_today(self) -> None:
        today = datetime.date.today()
        expected = today.year + 1 if today.month >= 10 else today.year
        assert current_season() == expected
