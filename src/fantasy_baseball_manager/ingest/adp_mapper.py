import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import httpx

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.protocols import ADPRepo, PlayerRepo

logger = logging.getLogger(__name__)

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

_FIXED_COLUMNS = {"Rank", "Player", "Team", "Positions", "AVG"}

# Lahman → modern abbreviation aliases (FantasyPros uses the modern style)
_TEAM_ALIASES: dict[str, str] = {
    "KCA": "KC",
    "TBA": "TB",
    "SFN": "SF",
    "SDN": "SD",
    "SLN": "STL",
    "CHN": "CHC",
    "CHA": "CWS",
    "LAN": "LAD",
    "NYA": "NYY",
    "NYN": "NYM",
    "WAS": "WSH",
    "ANA": "LAA",
}

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
_INITIAL_DOT_RE = re.compile(r"(?<!\w)([A-Za-z])\.")
_ADJACENT_INITIALS_RE = re.compile(r"(?<=\b[A-Za-z]) (?=[A-Za-z]\b)")

# Formal name → common short form used in baseball databases.
# Applied during normalization so "Matthew Boyd" matches "Matt Boyd".
_NICK_ALIASES: dict[str, str] = {
    "matthew": "matt",
    "michael": "mike",
    "christopher": "chris",
    "nicholas": "nick",
    "alexander": "alex",
    "benjamin": "ben",
    "gregory": "greg",
    "timothy": "tim",
    "stephen": "steve",
    "steven": "steve",
    "jeffrey": "jeff",
    "zachary": "zach",
    "frederick": "fred",
    "nathaniel": "nate",
    "jonathan": "jon",
    "abraham": "abe",
}


def _normalize_name(name: str) -> str:
    name = _PARENTHETICAL_RE.sub("", name)
    name = _SUFFIX_RE.sub("", name)
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Strip periods from initials: "J. T." -> "J  T " / "J.T." -> "JT "
    stripped = _INITIAL_DOT_RE.sub(r"\1", stripped)
    # Collapse whitespace, then merge adjacent single-letter tokens: "J T" -> "JT"
    stripped = " ".join(stripped.split())
    stripped = _ADJACENT_INITIALS_RE.sub("", stripped)
    lowered = stripped.strip().lower()
    # Apply nickname aliases to each token
    tokens = lowered.split()
    tokens = [_NICK_ALIASES.get(t, t) for t in tokens]
    return " ".join(tokens)


def fetch_mlb_active_teams(season: int) -> dict[int, str]:
    """Fetch mlbam_id → team abbreviation for all active MLB players from the MLB API."""
    with httpx.Client(timeout=30) as client:
        teams_resp = client.get(f"{_MLB_API_BASE}/teams", params={"sportId": 1, "season": season})
        teams_resp.raise_for_status()
        team_map: dict[int, str] = {}
        for t in teams_resp.json().get("teams", []):
            team_map[t["id"]] = t["abbreviation"]

        players_resp = client.get(f"{_MLB_API_BASE}/sports/1/players", params={"season": season})
        players_resp.raise_for_status()
        result: dict[int, str] = {}
        for p in players_resp.json().get("people", []):
            team_id = p.get("currentTeam", {}).get("id")
            abbrev = team_map.get(team_id, "") if team_id else ""
            if abbrev:
                result[p["id"]] = abbrev
    logger.debug("Fetched %d active player-team mappings from MLB API", len(result))
    return result


def _build_player_lookups(
    players: list[Player],
    player_teams: dict[int, str] | None = None,
) -> tuple[dict[tuple[str, str], int], dict[str, list[int]]]:
    by_name_team: dict[tuple[str, str], int] = {}
    by_name: dict[str, list[int]] = {}

    for p in players:
        if p.id is None:
            continue
        full_name = f"{p.name_first} {p.name_last}"
        normalized = _normalize_name(full_name)
        by_name.setdefault(normalized, []).append(p.id)
        if player_teams and p.id in player_teams:
            raw_team = player_teams[p.id]
            team_abbrev = _TEAM_ALIASES.get(raw_team, raw_team)
            by_name_team[(normalized, team_abbrev)] = p.id

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


def _split_raw_name(raw_name: str) -> tuple[str, str]:
    """Split a raw ADP player name into (first, last) for stub creation."""
    name = _PARENTHETICAL_RE.sub("", raw_name).strip()
    name = _SUFFIX_RE.sub("", name).strip()
    parts = name.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


@dataclass(frozen=True)
class ADPIngestResult:
    loaded: int
    skipped: int
    unmatched: list[str]
    created: int = 0


def ingest_fantasypros_adp(
    rows: list[dict[str, Any]],
    repo: ADPRepo,
    players: list[Player],
    season: int,
    as_of: str | None = None,
    player_teams: dict[int, str] | None = None,
    player_repo: PlayerRepo | None = None,
) -> ADPIngestResult:
    by_name_team, by_name = _build_player_lookups(players, player_teams)

    if not rows:
        return ADPIngestResult(loaded=0, skipped=0, unmatched=[])

    header = list(rows[0].keys())
    provider_columns = _discover_provider_columns(header)

    loaded = 0
    skipped = 0
    created = 0
    unmatched: list[str] = []

    for row in rows:
        player_id = _resolve_player(row, by_name_team, by_name)
        if player_id is None:
            raw_name = row.get("Player") or ""
            normalized = _normalize_name(raw_name)
            candidates = by_name.get(normalized, [])
            # Only create stubs for truly missing players (not ambiguous ones)
            if player_repo is not None and len(candidates) == 0:
                first, last = _split_raw_name(raw_name)
                player_id = player_repo.upsert(Player(name_first=first, name_last=last))
                by_name.setdefault(normalized, []).append(player_id)
                created += 1
                logger.debug("Created stub player '%s %s' (id=%d)", first, last, player_id)
            else:
                unmatched.append(raw_name or "???")
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

    return ADPIngestResult(loaded=loaded, skipped=skipped, unmatched=unmatched, created=created)
