from datetime import date

from fantasy_baseball_manager.statcast.calendar import game_dates, season_date_range


class TestSeasonDateRange:
    def test_returns_start_and_end_for_regular_season(self) -> None:
        start, end = season_date_range(2024)
        assert start == date(2024, 3, 20)
        assert end == date(2024, 11, 5)

    def test_returns_start_and_end_for_2020_covid_season(self) -> None:
        start, end = season_date_range(2020)
        assert start == date(2020, 7, 23)
        assert end == date(2020, 10, 28)

    def test_default_season_bounds(self) -> None:
        start, end = season_date_range(2019)
        assert start == date(2019, 3, 20)
        assert end == date(2019, 11, 5)


class TestGameDates:
    def test_returns_list_of_dates(self) -> None:
        dates = game_dates(2024)
        assert isinstance(dates, list)
        assert all(isinstance(d, date) for d in dates)

    def test_dates_are_within_season_range(self) -> None:
        start, end = season_date_range(2024)
        dates = game_dates(2024)
        assert dates[0] == start
        assert dates[-1] == end

    def test_dates_are_sorted(self) -> None:
        dates = game_dates(2024)
        assert dates == sorted(dates)

    def test_no_duplicates(self) -> None:
        dates = game_dates(2024)
        assert len(dates) == len(set(dates))

    def test_2020_shorter_season(self) -> None:
        dates_2020 = game_dates(2020)
        dates_2024 = game_dates(2024)
        assert len(dates_2020) < len(dates_2024)
