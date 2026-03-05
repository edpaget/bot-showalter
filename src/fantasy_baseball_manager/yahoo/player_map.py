import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import YahooPlayerMap
from fantasy_baseball_manager.name_utils import normalize_name, strip_accents, strip_name_decorations

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Player
    from fantasy_baseball_manager.repos import PlayerRepo, YahooPlayerMapRepo
logger = logging.getLogger(__name__)

_PITCHING_POSITIONS = {"SP", "RP", "P"}


class YahooPlayerMapper:
    def __init__(
        self,
        map_repo: YahooPlayerMapRepo,
        player_repo: PlayerRepo,
    ) -> None:
        self._map_repo = map_repo
        self._player_repo = player_repo

    def resolve(self, yahoo_player_data: dict[str, Any]) -> YahooPlayerMap | None:
        yahoo_key: str = yahoo_player_data["player_key"]
        name: str = yahoo_player_data.get("name", "")
        team: str = yahoo_player_data.get("editorial_team_abbr", "")
        positions: list[str] = yahoo_player_data.get("eligible_positions", [])

        # Strategy 1: exact key lookup
        existing = self._map_repo.get_by_yahoo_key(yahoo_key)
        if existing is not None:
            return existing

        player_type = self.infer_player_type(positions)
        positions_str = ",".join(positions)

        # Strategy 2: MLBAM ID from Yahoo metadata
        mlbam_id = yahoo_player_data.get("player_id")
        if mlbam_id is not None:
            player = self._player_repo.get_by_mlbam_id(int(mlbam_id))
            if player is not None and player.id is not None:
                mapping = YahooPlayerMap(
                    yahoo_player_key=yahoo_key,
                    player_id=player.id,
                    player_type=player_type,
                    yahoo_name=name,
                    yahoo_team=team,
                    yahoo_positions=positions_str,
                )
                self._map_repo.upsert(mapping)
                return self._map_repo.get_by_yahoo_key(yahoo_key)

        # Strategy 3: progressive last-name search with normalized fallback
        if name:
            clean_name = strip_name_decorations(name)
            parts = clean_name.split()
            if len(parts) >= 2:
                normalized_yahoo = normalize_name(name)

                # Progressive exact last-name search: "De La Cruz" → "La Cruz" → "Cruz"
                last_name_suffixes = [" ".join(parts[i:]) for i in range(1, len(parts))]
                for suffix in last_name_suffixes:
                    candidates = self._player_repo.get_by_last_name(suffix)
                    matched = self._match_candidates(candidates, normalized_yahoo)
                    if matched is not None:
                        return self._persist(matched, yahoo_key, player_type, name, team, positions_str)

                # Retry with accent-normalized search
                for suffix in last_name_suffixes:
                    stripped_suffix = strip_accents(suffix)
                    candidates = self._player_repo.search_by_last_name_normalized(stripped_suffix)
                    matched = self._match_candidates(candidates, normalized_yahoo)
                    if matched is not None:
                        return self._persist(matched, yahoo_key, player_type, name, team, positions_str)

                # Broad search_by_name fallback
                candidates = self._player_repo.search_by_name(strip_accents(clean_name))
                matched = self._match_candidates(candidates, normalized_yahoo)
                if matched is not None:
                    return self._persist(matched, yahoo_key, player_type, name, team, positions_str)

        logger.warning(
            "Could not resolve Yahoo player: key=%s name=%s team=%s positions=%s",
            yahoo_key,
            name,
            team,
            positions_str,
        )
        return None

    def _match_candidates(self, candidates: list[Player], normalized_yahoo: str) -> Player | None:
        """Find a single matching player from candidates by normalized name or uniqueness."""
        for candidate in candidates:
            if candidate.name_first and candidate.id is not None:
                candidate_full = f"{candidate.name_first} {candidate.name_last}"
                if normalize_name(candidate_full) == normalized_yahoo:
                    return candidate
        if len(candidates) == 1 and candidates[0].id is not None:
            return candidates[0]
        return None

    def _persist(
        self,
        player: Player,
        yahoo_key: str,
        player_type: str,
        name: str,
        team: str,
        positions_str: str,
    ) -> YahooPlayerMap | None:
        """Create the mapping, persist it, and return the persisted copy."""
        mapping = YahooPlayerMap(
            yahoo_player_key=yahoo_key,
            player_id=player.id,  # type: ignore[arg-type]
            player_type=player_type,
            yahoo_name=name,
            yahoo_team=team,
            yahoo_positions=positions_str,
        )
        self._map_repo.upsert(mapping)
        return self._map_repo.get_by_yahoo_key(yahoo_key)

    @staticmethod
    def infer_player_type(positions: list[str]) -> str:
        filtered = [p for p in positions if p not in ("Util", "BN", "IL", "IL+", "NA", "DL")]
        if not filtered:
            return "batter"
        if all(p in _PITCHING_POSITIONS for p in filtered):
            return "pitcher"
        return "batter"
