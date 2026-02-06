"""DataSource implementation for pitch sequence data.

Provides PitchSequenceDataSource which implements DataSource[PlayerContext],
building pitch sequences for individual players via GameSequenceBuilder.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, overload

from fantasy_baseball_manager.context import get_context
from fantasy_baseball_manager.contextual.data.models import PlayerContext
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.result import Err, Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
    from fantasy_baseball_manager.contextual.data.cache import SequenceCache
    from fantasy_baseball_manager.player.identity import Player

logger = logging.getLogger(__name__)


class PitchSequenceDataSource:
    """DataSource[PlayerContext] for pitch-level event sequences.

    Builds pitch sequences for individual players. Supports single-player
    and list queries. ALL_PLAYERS is rejected as too expensive.
    """

    def __init__(
        self,
        builder: GameSequenceBuilder,
        perspective: str = "batter",
        game_window: int | None = None,
        cache: SequenceCache | None = None,
    ) -> None:
        self._builder = builder
        self._perspective = perspective
        self._game_window = game_window
        self._cache = cache

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[PlayerContext]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[PlayerContext]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[PlayerContext] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[PlayerContext]] | Ok[PlayerContext] | Err[DataSourceError]:
        if query is ALL_PLAYERS:
            return Err(DataSourceError("ALL_PLAYERS queries not supported for pitch sequences"))

        if isinstance(query, list):
            return self._query_list(query)

        return self._query_single(query)

    def _query_single(self, player: Player) -> Ok[PlayerContext] | Err[DataSourceError]:
        if player.mlbam_id is None:
            return Err(DataSourceError(f"Player {player.name} has no mlbam_id"))

        season = get_context().year
        player_id = int(player.mlbam_id)

        # Check cache
        if self._cache is not None:
            cached = self._cache.get(season, player_id, self._perspective)
            if cached is not None:
                return Ok(self._apply_window(cached))

        try:
            games = self._builder.build_player_season(season, player_id, perspective=self._perspective)
        except Exception as e:
            return Err(DataSourceError(f"Failed to build sequences for {player.name}", e))

        ctx = PlayerContext(
            player_id=player_id,
            player_name=player.name,
            season=season,
            perspective=self._perspective,
            games=tuple(games),
        )

        if self._cache is not None:
            self._cache.put(ctx)

        return Ok(self._apply_window(ctx))

    def _query_list(self, players: list[Player]) -> Ok[list[PlayerContext]] | Err[DataSourceError]:
        results: list[PlayerContext] = []
        for player in players:
            result = self._query_single(player)
            if result.is_err():
                err = result.unwrap_err()
                if not isinstance(err, DataSourceError):
                    err = DataSourceError(str(err))
                return Err(err)
            results.append(result.unwrap())
        return Ok(results)

    def _apply_window(self, ctx: PlayerContext) -> PlayerContext:
        if self._game_window is None or len(ctx.games) <= self._game_window:
            return ctx
        return PlayerContext(
            player_id=ctx.player_id,
            player_name=ctx.player_name,
            season=ctx.season,
            perspective=ctx.perspective,
            games=ctx.games[-self._game_window :],
        )
