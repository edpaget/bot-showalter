import logging

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.valuation import PlayerValuation
from fantasy_baseball_manager.repos.protocols import PlayerRepo, ValuationRepo

logger = logging.getLogger(__name__)


class ValuationLookupService:
    def __init__(self, player_repo: PlayerRepo, valuation_repo: ValuationRepo) -> None:
        self._player_repo = player_repo
        self._valuation_repo = valuation_repo

    def lookup(self, player_name: str, season: int, system: str | None = None) -> list[PlayerValuation]:
        logger.debug("Valuation lookup: player=%s season=%d system=%s", player_name, season, system)
        if "," in player_name:
            last, _, first = player_name.partition(",")
            last = last.strip()
            first = first.strip()
        else:
            last = player_name.strip()
            first = None

        players = self._player_repo.get_by_last_name(last)

        if first:
            players = [p for p in players if p.name_first and p.name_first.lower() == first.lower()]

        results: list[PlayerValuation] = []
        for player in players:
            assert player.id is not None
            valuations = self._valuation_repo.get_by_player_season(player.id, season, system)
            for val in valuations:
                results.append(
                    PlayerValuation(
                        player_name=f"{player.name_first} {player.name_last}",
                        system=val.system,
                        version=val.version,
                        projection_system=val.projection_system,
                        projection_version=val.projection_version,
                        player_type=val.player_type,
                        position=val.position,
                        value=val.value,
                        rank=val.rank,
                        category_scores=val.category_scores,
                    )
                )

        logger.debug("Valuation lookup returned %d results", len(results))
        return results

    def rankings(
        self,
        season: int,
        system: str | None = None,
        player_type: str | None = None,
        position: str | None = None,
        top: int | None = None,
    ) -> list[PlayerValuation]:
        valuations = self._valuation_repo.get_by_season(season, system)

        if player_type is not None:
            valuations = [v for v in valuations if v.player_type == player_type]

        if position is not None:
            valuations = [v for v in valuations if v.position == position]

        valuations.sort(key=lambda v: v.rank)

        if top is not None:
            valuations = valuations[:top]

        player_ids = [val.player_id for val in valuations]
        players = self._player_repo.get_by_ids(player_ids)
        player_map: dict[int, Player] = {p.id: p for p in players if p.id is not None}

        results: list[PlayerValuation] = []
        for val in valuations:
            player = player_map.get(val.player_id)
            player_display = f"{player.name_first} {player.name_last}" if player else f"Unknown ({val.player_id})"
            results.append(
                PlayerValuation(
                    player_name=player_display,
                    system=val.system,
                    version=val.version,
                    projection_system=val.projection_system,
                    projection_version=val.projection_version,
                    player_type=val.player_type,
                    position=val.position,
                    value=val.value,
                    rank=val.rank,
                    category_scores=val.category_scores,
                )
            )

        return results
