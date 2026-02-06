from __future__ import annotations

from datetime import date, timedelta

# Season-specific overrides for non-standard start/end dates.
_SEASON_OVERRIDES: dict[int, tuple[date, date]] = {
    2020: (date(2020, 7, 23), date(2020, 10, 28)),
}

_DEFAULT_START_MONTH_DAY = (3, 20)
_DEFAULT_END_MONTH_DAY = (11, 5)


def season_date_range(season: int) -> tuple[date, date]:
    """Return the (start, end) dates for an MLB season, inclusive."""
    if season in _SEASON_OVERRIDES:
        return _SEASON_OVERRIDES[season]
    start = date(season, *_DEFAULT_START_MONTH_DAY)
    end = date(season, *_DEFAULT_END_MONTH_DAY)
    return start, end


def game_dates(season: int) -> list[date]:
    """Return every date in the season range, inclusive."""
    start, end = season_date_range(season)
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days
