import datetime

_OCTOBER = 10


def current_season(today: datetime.date | None = None) -> int:
    """Return the current or upcoming baseball season.

    From January through September the current calendar year is the active
    season.  Starting in October the fantasy community shifts focus to next
    year's draft, so we return the *upcoming* season instead.
    """
    if today is None:
        today = datetime.date.today()
    return today.year + 1 if today.month >= _OCTOBER else today.year
