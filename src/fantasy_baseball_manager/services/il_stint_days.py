from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import ILStint

# Default days for IL types when no days or dates are available
IL_TYPE_DEFAULTS: dict[str, int] = {
    "10-day": 10,
    "15-day": 15,
    "60-day": 60,
    "7-day": 7,
}


def compute_stint_days(stint: ILStint) -> int:
    """Compute days lost for a stint: use days field, then date diff, then IL type default."""
    if stint.days is not None:
        return stint.days
    if stint.end_date is not None:
        try:
            start = date.fromisoformat(stint.start_date)
            end = date.fromisoformat(stint.end_date)
            diff = (end - start).days
            return max(diff, 0)
        except ValueError:
            pass
    return IL_TYPE_DEFAULTS.get(stint.il_type, 15)
