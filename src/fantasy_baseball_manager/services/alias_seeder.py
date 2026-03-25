from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PlayerAlias, PlayerType
from fantasy_baseball_manager.name_utils import normalize_name

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import ConnectionProvider, PlayerAliasRepo, PlayerRepo


def seed_aliases(
    player_repo: PlayerRepo,
    alias_repo: PlayerAliasRepo,
    conn_provider: ConnectionProvider,
) -> int:
    """Populate the player_alias table from existing player and stats data.

    Returns the number of aliases upserted.
    """
    players = player_repo.all()
    batting_ranges = _get_active_ranges(conn_provider, "batting_stats")
    pitching_ranges = _get_active_ranges(conn_provider, "pitching_stats")

    aliases: list[PlayerAlias] = []
    for player in players:
        if player.id is None:
            continue
        pid = player.id

        is_batter = pid in batting_ranges
        is_pitcher = pid in pitching_ranges

        if not is_batter and not is_pitcher:
            continue

        active_from, active_to = _merge_ranges(
            batting_ranges.get(pid),
            pitching_ranges.get(pid),
        )

        name_variants = _generate_name_variants(player.name_first, player.name_last)

        if is_batter and is_pitcher:
            # Two-way player: separate aliases per type
            for name in name_variants:
                b_from, b_to = batting_ranges[pid]
                aliases.append(
                    PlayerAlias(
                        alias_name=name,
                        player_id=pid,
                        player_type=PlayerType.BATTER,
                        source="seed",
                        active_from=b_from,
                        active_to=b_to,
                    )
                )
                p_from, p_to = pitching_ranges[pid]
                aliases.append(
                    PlayerAlias(
                        alias_name=name,
                        player_id=pid,
                        player_type=PlayerType.PITCHER,
                        source="seed",
                        active_from=p_from,
                        active_to=p_to,
                    )
                )
        else:
            ptype = PlayerType.BATTER if is_batter else PlayerType.PITCHER
            for name in name_variants:
                aliases.append(
                    PlayerAlias(
                        alias_name=name,
                        player_id=pid,
                        player_type=ptype,
                        source="seed",
                        active_from=active_from,
                        active_to=active_to,
                    )
                )

    return alias_repo.upsert_batch(aliases)


_ALLOWED_STATS_TABLES = {"batting_stats", "pitching_stats"}


def _get_active_ranges(
    provider: ConnectionProvider,
    table: str,
) -> dict[int, tuple[int, int]]:
    """Return {player_id: (min_season, max_season)} from a stats table."""
    if table not in _ALLOWED_STATS_TABLES:
        msg = f"Disallowed table: {table}"
        raise ValueError(msg)
    with provider.connection() as conn:
        rows = conn.execute(
            f"SELECT player_id, MIN(season), MAX(season) FROM {table} GROUP BY player_id"  # noqa: S608
        ).fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}


def _merge_ranges(
    batting: tuple[int, int] | None,
    pitching: tuple[int, int] | None,
) -> tuple[int, int]:
    """Merge batting and pitching active ranges into a single span."""
    if batting and pitching:
        return (min(batting[0], pitching[0]), max(batting[1], pitching[1]))
    if batting:
        return batting
    if pitching:
        return pitching
    msg = "At least one range must be provided"
    raise ValueError(msg)


def _generate_name_variants(first: str, last: str) -> set[str]:
    """Generate normalized name variants for a player."""
    variants: set[str] = set()

    # "first last" with nick aliases (e.g. "matt boyd")
    primary = normalize_name(f"{first} {last}")
    variants.add(primary)

    # Formal form without nick aliases (e.g. "matthew boyd")
    formal = normalize_name(f"{first} {last}", apply_nicks=False)
    if formal != primary:
        variants.add(formal)

    # "last, first" normalized (e.g. "boyd matt" after comma removal)
    variants.add(normalize_name(f"{last}, {first}"))

    # "F last" first-initial variant (e.g. "m trout")
    if len(first) > 1:
        variants.add(normalize_name(f"{first[0]} {last}"))

    return variants
