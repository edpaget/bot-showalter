"""File-based parquet cache for pitch sequence data.

Stores flattened pitch data as parquet files, one per player/season/perspective.
Reconstructs typed GameSequence and PlayerContext objects on read.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

from fantasy_baseball_manager.contextual.data.models import (
    GameSequence,
    PitchEvent,
    PlayerContext,
)


class SequenceCache:
    """Parquet-based cache for PlayerContext objects.

    File layout: {cache_dir}/sequences/{season}/{perspective}/{player_id}.parquet

    Stores one row per pitch with game metadata denormalized. Reconstructs
    typed objects on read by grouping on game_pk.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def _player_path(self, season: int, player_id: int, perspective: str) -> Path:
        return self._cache_dir / "sequences" / str(season) / perspective / f"{player_id}.parquet"

    def _season_dir(self, season: int) -> Path:
        return self._cache_dir / "sequences" / str(season)

    def has(self, season: int, player_id: int, perspective: str) -> bool:
        """Check if cached data exists for a player/season/perspective."""
        return self._player_path(season, player_id, perspective).exists()

    def get(self, season: int, player_id: int, perspective: str) -> PlayerContext | None:
        """Read cached data, returning None on cache miss."""
        path = self._player_path(season, player_id, perspective)
        if not path.exists():
            return None

        df = pd.read_parquet(path)
        return self._df_to_context(df, player_id, season, perspective)

    def put(self, context: PlayerContext) -> None:
        """Write a PlayerContext to cache as a flat parquet table."""
        rows: list[dict[str, object]] = []
        for game in context.games:
            for pitch in game.pitches:
                row: dict[str, object] = {
                    # Game metadata
                    "game_pk": game.game_pk,
                    "game_date": game.game_date,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    # Player metadata
                    "player_name": context.player_name,
                    # Pitch fields
                    "batter_id": pitch.batter_id,
                    "pitcher_id": pitch.pitcher_id,
                    "pitch_type": pitch.pitch_type,
                    "pitch_result": pitch.pitch_result,
                    "pitch_result_type": pitch.pitch_result_type,
                    "release_speed": pitch.release_speed,
                    "release_spin_rate": pitch.release_spin_rate,
                    "pfx_x": pitch.pfx_x,
                    "pfx_z": pitch.pfx_z,
                    "plate_x": pitch.plate_x,
                    "plate_z": pitch.plate_z,
                    "release_extension": pitch.release_extension,
                    "launch_speed": pitch.launch_speed,
                    "launch_angle": pitch.launch_angle,
                    "hit_distance": pitch.hit_distance,
                    "bb_type": pitch.bb_type,
                    "estimated_woba": pitch.estimated_woba,
                    "inning": pitch.inning,
                    "is_top": pitch.is_top,
                    "outs": pitch.outs,
                    "balls": pitch.balls,
                    "strikes": pitch.strikes,
                    "runners_on_1b": pitch.runners_on_1b,
                    "runners_on_2b": pitch.runners_on_2b,
                    "runners_on_3b": pitch.runners_on_3b,
                    "bat_score": pitch.bat_score,
                    "fld_score": pitch.fld_score,
                    "stand": pitch.stand,
                    "p_throws": pitch.p_throws,
                    "pitch_number": pitch.pitch_number,
                    "pa_event": pitch.pa_event,
                    "delta_run_exp": pitch.delta_run_exp,
                    "at_bat_number": pitch.at_bat_number,
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        path = self._player_path(context.season, context.player_id, context.perspective)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def invalidate(
        self,
        season: int,
        player_id: int | None = None,
        perspective: str | None = None,
    ) -> None:
        """Remove cached data.

        If player_id and perspective are given, removes one file.
        If only season is given, removes the entire season directory.
        """
        if player_id is not None and perspective is not None:
            path = self._player_path(season, player_id, perspective)
            if path.exists():
                path.unlink()
        else:
            season_dir = self._season_dir(season)
            if season_dir.exists():
                shutil.rmtree(season_dir)

    def _df_to_context(self, df: pd.DataFrame, player_id: int, season: int, perspective: str) -> PlayerContext:
        """Reconstruct a PlayerContext from a flat DataFrame."""
        player_name = str(df.iloc[0]["player_name"])
        games: list[GameSequence] = []

        for game_pk, group in df.groupby("game_pk", sort=False):
            pitches: list[PitchEvent] = []
            for row in group.itertuples(index=False):
                pitch = PitchEvent(
                    batter_id=int(row.batter_id),
                    pitcher_id=int(row.pitcher_id),
                    pitch_type=str(row.pitch_type),
                    pitch_result=str(row.pitch_result),
                    pitch_result_type=str(row.pitch_result_type),
                    release_speed=_opt_float(row.release_speed),
                    release_spin_rate=_opt_int(row.release_spin_rate),
                    pfx_x=_opt_float(row.pfx_x),
                    pfx_z=_opt_float(row.pfx_z),
                    plate_x=_opt_float(row.plate_x),
                    plate_z=_opt_float(row.plate_z),
                    release_extension=_opt_float(row.release_extension),
                    launch_speed=_opt_float(row.launch_speed),
                    launch_angle=_opt_int(row.launch_angle),
                    hit_distance=_opt_int(row.hit_distance),
                    bb_type=_opt_str(row.bb_type),
                    estimated_woba=_opt_float(row.estimated_woba),
                    inning=int(row.inning),
                    is_top=bool(row.is_top),
                    outs=int(row.outs),
                    balls=int(row.balls),
                    strikes=int(row.strikes),
                    runners_on_1b=bool(row.runners_on_1b),
                    runners_on_2b=bool(row.runners_on_2b),
                    runners_on_3b=bool(row.runners_on_3b),
                    bat_score=int(row.bat_score),
                    fld_score=int(row.fld_score),
                    stand=str(row.stand),
                    p_throws=str(row.p_throws),
                    pitch_number=int(row.pitch_number),
                    pa_event=_opt_str(row.pa_event),
                    delta_run_exp=_opt_float(row.delta_run_exp),
                    at_bat_number=_opt_int(row.at_bat_number) if hasattr(row, "at_bat_number") else None,
                )
                pitches.append(pitch)

            first = group.iloc[0]
            games.append(
                GameSequence(
                    game_pk=int(game_pk),
                    game_date=str(first["game_date"]),
                    season=season,
                    home_team=str(first["home_team"]),
                    away_team=str(first["away_team"]),
                    perspective=perspective,
                    player_id=player_id,
                    pitches=tuple(pitches),
                )
            )

        return PlayerContext(
            player_id=player_id,
            player_name=player_name,
            season=season,
            perspective=perspective,
            games=tuple(games),
        )


def _opt_float(val: object) -> float | None:
    if pd.isna(val):
        return None
    return float(val)  # type: ignore[arg-type]


def _opt_int(val: object) -> int | None:
    if pd.isna(val):
        return None
    return int(val)  # type: ignore[arg-type]


def _opt_str(val: object) -> str | None:
    if pd.isna(val):
        return None
    return str(val)
