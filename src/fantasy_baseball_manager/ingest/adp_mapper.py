import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.protocols import ADPRepo

logger = logging.getLogger(__name__)

_FIXED_COLUMNS = {"Rank", "Player", "Team", "Positions", "AVG"}

_PROVIDER_SLUGS: dict[str, str] = {
    "ESPN": "espn",
    "Yahoo": "yahoo",
    "CBS": "cbs",
    "NFBC": "nfbc",
    "RTS": "rts",
    "FT": "ft",
}

_SUFFIX_RE = re.compile(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)\s*$", re.IGNORECASE)
_PARENTHETICAL_RE = re.compile(r"\s*\((?:Batter|Pitcher)\)\s*$", re.IGNORECASE)


def _normalize_name(name: str) -> str:
    name = _PARENTHETICAL_RE.sub("", name)
    name = _SUFFIX_RE.sub("", name)
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.strip().lower()


def _build_player_lookups(
    players: list[Player],
) -> tuple[dict[tuple[str, str], int], dict[str, list[int]]]:
    by_name_team: dict[tuple[str, str], int] = {}
    by_name: dict[str, list[int]] = {}

    for p in players:
        if p.id is None:
            continue
        full_name = f"{p.name_first} {p.name_last}"
        normalized = _normalize_name(full_name)
        by_name.setdefault(normalized, []).append(p.id)

    for p in players:
        if p.id is None:
            continue
        full_name = f"{p.name_first} {p.name_last}"
        normalized = _normalize_name(full_name)
        # We don't have team abbreviations on Player, so name-only lookup is primary.
        # The by_name_team lookup is populated from roster data if available;
        # for now we skip it and rely on name-only with disambiguation.

    return by_name_team, by_name


def _resolve_player(
    row: dict[str, Any],
    by_name_team: dict[tuple[str, str], int],
    by_name: dict[str, list[int]],
) -> int | None:
    raw_name = row.get("Player") or ""
    team = (row.get("Team") or "").strip()
    normalized = _normalize_name(raw_name)

    if team:
        player_id = by_name_team.get((normalized, team))
        if player_id is not None:
            return player_id

    candidates = by_name.get(normalized, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.debug("Ambiguous player name '%s' (%d matches), skipping", raw_name, len(candidates))
        return None

    logger.debug("No player match for '%s' (team=%s)", raw_name, team)
    return None


def _discover_provider_columns(header: list[str]) -> list[tuple[str, str]]:
    providers: list[tuple[str, str]] = []
    for col in header:
        if col in _FIXED_COLUMNS:
            continue
        slug = _PROVIDER_SLUGS.get(col)
        if slug is not None:
            providers.append((col, slug))
    return providers


@dataclass(frozen=True)
class ADPIngestResult:
    loaded: int
    skipped: int
    unmatched: list[str]


def ingest_fantasypros_adp(
    rows: list[dict[str, Any]],
    repo: ADPRepo,
    players: list[Player],
    season: int,
    as_of: str | None = None,
) -> ADPIngestResult:
    by_name_team, by_name = _build_player_lookups(players)

    if not rows:
        return ADPIngestResult(loaded=0, skipped=0, unmatched=[])

    header = list(rows[0].keys())
    provider_columns = _discover_provider_columns(header)

    loaded = 0
    skipped = 0
    unmatched: list[str] = []

    for row in rows:
        player_id = _resolve_player(row, by_name_team, by_name)
        if player_id is None:
            unmatched.append(row.get("Player", "???"))
            continue

        rank_str = row.get("Rank", "").strip()
        if not rank_str:
            skipped += 1
            continue
        rank = int(rank_str)
        positions = row.get("Positions", "").strip()

        avg_str = row.get("AVG", "").strip().replace(",", "")
        if avg_str:
            repo.upsert(
                ADP(
                    player_id=player_id,
                    season=season,
                    provider="fantasypros",
                    overall_pick=float(avg_str),
                    rank=rank,
                    positions=positions,
                    as_of=as_of,
                )
            )
            loaded += 1

        for col_name, slug in provider_columns:
            val = row.get(col_name, "").strip().replace(",", "")
            if not val:
                continue
            repo.upsert(
                ADP(
                    player_id=player_id,
                    season=season,
                    provider=slug,
                    overall_pick=float(val),
                    rank=rank,
                    positions=positions,
                    as_of=as_of,
                )
            )
            loaded += 1

    return ADPIngestResult(loaded=loaded, skipped=skipped, unmatched=unmatched)
