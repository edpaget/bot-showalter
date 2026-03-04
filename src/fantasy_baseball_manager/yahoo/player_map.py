import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import YahooPlayerMap

if TYPE_CHECKING:
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

        # Strategy 3: name search
        if name:
            parts = name.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                first_name = parts[0]
                candidates = self._player_repo.get_by_last_name(last_name)
                # Try exact first+last match
                for candidate in candidates:
                    if (
                        candidate.name_first
                        and candidate.name_first.lower() == first_name.lower()
                        and candidate.id is not None
                    ):
                        mapping = YahooPlayerMap(
                            yahoo_player_key=yahoo_key,
                            player_id=candidate.id,
                            player_type=player_type,
                            yahoo_name=name,
                            yahoo_team=team,
                            yahoo_positions=positions_str,
                        )
                        self._map_repo.upsert(mapping)
                        return self._map_repo.get_by_yahoo_key(yahoo_key)
                # Single last-name match as fallback
                if len(candidates) == 1 and candidates[0].id is not None:
                    mapping = YahooPlayerMap(
                        yahoo_player_key=yahoo_key,
                        player_id=candidates[0].id,
                        player_type=player_type,
                        yahoo_name=name,
                        yahoo_team=team,
                        yahoo_positions=positions_str,
                    )
                    self._map_repo.upsert(mapping)
                    return self._map_repo.get_by_yahoo_key(yahoo_key)

        logger.warning(
            "Could not resolve Yahoo player: key=%s name=%s team=%s positions=%s",
            yahoo_key,
            name,
            team,
            positions_str,
        )
        return None

    @staticmethod
    def infer_player_type(positions: list[str]) -> str:
        filtered = [p for p in positions if p not in ("Util", "BN", "IL", "IL+", "NA", "DL")]
        if not filtered:
            return "batter"
        if all(p in _PITCHING_POSITIONS for p in filtered):
            return "pitcher"
        return "batter"
