#!/usr/bin/env python
"""Generate consensus playing-time projections by averaging Steamer and ZiPS.

Stores results in the projection table as system='playing_time', version='latest'.
This provides playing-time estimates for the composite model's rate-to-counting
stat conversion.

Usage:
    uv run scripts/generate_consensus_pt.py [--seasons 2021 2022 2023 2024 2025]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DEFAULT_SEASONS = list(range(2021, 2026))
DB_PATH = Path("data/fbm.db")


def generate_consensus_pt(conn: sqlite3.Connection, seasons: list[int]) -> None:
    """Generate consensus PT from Steamer+ZiPS averages for given seasons."""
    # Clear existing playing_time/latest for these seasons
    placeholders = ", ".join("?" for _ in seasons)
    conn.execute(
        f"DELETE FROM projection WHERE system = 'playing_time' AND version = 'latest' AND season IN ({placeholders})",
        seasons,
    )

    for season in seasons:
        version = str(season)

        # Batters: consensus PA
        conn.execute(
            """
            INSERT OR REPLACE INTO projection (player_id, season, system, version, player_type, pa, source_type)
            SELECT
                COALESCE(s.player_id, z.player_id) AS player_id,
                ? AS season,
                'playing_time' AS system,
                'latest' AS version,
                'batter' AS player_type,
                CASE
                    WHEN s.pa IS NOT NULL AND z.pa IS NOT NULL THEN CAST((s.pa + z.pa) / 2.0 AS INTEGER)
                    WHEN s.pa IS NOT NULL THEN s.pa
                    ELSE z.pa
                END AS pa,
                'first_party' AS source_type
            FROM (
                SELECT player_id, pa FROM projection
                WHERE system = 'steamer' AND version = ? AND player_type = 'batter' AND pa IS NOT NULL
            ) s
            FULL OUTER JOIN (
                SELECT player_id, pa FROM projection
                WHERE system = 'zips' AND version = ? AND player_type = 'batter' AND pa IS NOT NULL
            ) z ON s.player_id = z.player_id
            """,
            (season, version, version),
        )
        bat_count = conn.execute(
            "SELECT COUNT(*) FROM projection"
            " WHERE system = 'playing_time' AND version = 'latest'"
            " AND season = ? AND player_type = 'batter'",
            (season,),
        ).fetchone()[0]

        # Pitchers: consensus IP
        conn.execute(
            """
            INSERT OR REPLACE INTO projection (player_id, season, system, version, player_type, ip, source_type)
            SELECT
                COALESCE(s.player_id, z.player_id) AS player_id,
                ? AS season,
                'playing_time' AS system,
                'latest' AS version,
                'pitcher' AS player_type,
                CASE
                    WHEN s.ip IS NOT NULL AND z.ip IS NOT NULL THEN ROUND((s.ip + z.ip) / 2.0, 1)
                    WHEN s.ip IS NOT NULL THEN s.ip
                    ELSE z.ip
                END AS ip,
                'first_party' AS source_type
            FROM (
                SELECT player_id, ip FROM projection
                WHERE system = 'steamer' AND version = ? AND player_type = 'pitcher' AND ip IS NOT NULL
            ) s
            FULL OUTER JOIN (
                SELECT player_id, ip FROM projection
                WHERE system = 'zips' AND version = ? AND player_type = 'pitcher' AND ip IS NOT NULL
            ) z ON s.player_id = z.player_id
            """,
            (season, version, version),
        )
        pit_count = conn.execute(
            "SELECT COUNT(*) FROM projection"
            " WHERE system = 'playing_time' AND version = 'latest'"
            " AND season = ? AND player_type = 'pitcher'",
            (season,),
        ).fetchone()[0]

        print(f"season={season}: {bat_count} batters, {pit_count} pitchers")

    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consensus PT from Steamer+ZiPS")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS, help="Seasons to generate PT for")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database path")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    generate_consensus_pt(conn, args.seasons)
    conn.close()
    print("Done")


if __name__ == "__main__":
    main()
