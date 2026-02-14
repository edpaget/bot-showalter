from datetime import date, timedelta


def chunk_date_range(start_dt: str, end_dt: str, chunk_days: int = 7) -> list[tuple[str, str]]:
    """Split a date range into chunks of at most chunk_days days.

    Returns a list of (start, end) ISO date string pairs.
    """
    start = date.fromisoformat(start_dt)
    end = date.fromisoformat(end_dt)

    if start > end:
        raise ValueError(f"start_dt ({start_dt}) is after end_dt ({end_dt})")

    chunks: list[tuple[str, str]] = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)

    return chunks
