"""Builds GameSequence objects from raw Statcast DataFrames.

Transforms raw Statcast pitch-level data into typed, ordered GameSequence
objects grouped by (game_pk, player) from a given perspective.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from fantasy_baseball_manager.contextual.data.models import GameSequence, PitchEvent

if TYPE_CHECKING:
    from fantasy_baseball_manager.statcast.store import StatcastStore

logger = logging.getLogger(__name__)


def _opt_float(val: object) -> float | None:
    """Convert a value to float or None if NaN/missing."""
    if pd.isna(val):
        return None
    return float(val)  # type: ignore[arg-type]


def _opt_int(val: object) -> int | None:
    """Convert a value to int or None if NaN/missing."""
    if pd.isna(val):
        return None
    return int(val)  # type: ignore[arg-type]


def _opt_str(val: object) -> str | None:
    """Convert a value to str or None if NaN/missing."""
    if pd.isna(val):
        return None
    return str(val)


class GameSequenceBuilder:
    """Builds GameSequence objects from a StatcastStore.

    Uses dependency injection: accepts a StatcastStore at construction time
    rather than importing/instantiating one directly.
    """

    def __init__(self, store: StatcastStore) -> None:
        self._store = store

    def build_season(self, season: int, perspective: str = "batter") -> list[GameSequence]:
        """Build GameSequences for all players in a season.

        Args:
            season: The MLB season year.
            perspective: "batter" or "pitcher" — determines grouping column.

        Returns:
            List of GameSequence objects, one per (game, player) combination.
        """
        df = self._store.read_season(season)
        if df.empty:
            return []
        return self._build_from_df(df, season, perspective)

    def build_player_season(self, season: int, player_id: int, perspective: str = "batter") -> list[GameSequence]:
        """Build GameSequences for a single player in a season.

        More efficient than build_season + filter: pre-filters the DataFrame.

        Args:
            season: The MLB season year.
            player_id: MLBAM ID of the player.
            perspective: "batter" or "pitcher".

        Returns:
            List of GameSequence objects for the given player.
        """
        df = self._store.read_season(season)
        if df.empty:
            return []
        player_col = "batter" if perspective == "batter" else "pitcher"
        df = df[df[player_col] == player_id]
        if df.empty:
            return []
        return self._build_from_df(df, season, perspective)

    def _build_from_df(self, df: pd.DataFrame, season: int, perspective: str) -> list[GameSequence]:
        """Core logic: filter, sort, group, convert."""
        # Drop rows with null pitch_type
        df = df[df["pitch_type"].notna()]
        if df.empty:
            return []

        # Sort by game, at-bat, pitch number
        df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])

        player_col = "batter" if perspective == "batter" else "pitcher"
        sequences: list[GameSequence] = []

        for (game_pk, player_id_val), group in df.groupby(["game_pk", player_col]):
            pitches: list[PitchEvent] = []
            for row in group.itertuples(index=False):
                is_top = row.inning_topbot == "Top"
                if is_top:
                    bat_score = int(row.away_score)
                    fld_score = int(row.home_score)
                else:
                    bat_score = int(row.home_score)
                    fld_score = int(row.away_score)

                pitch = PitchEvent(
                    batter_id=int(row.batter),
                    pitcher_id=int(row.pitcher),
                    pitch_type=str(row.pitch_type),
                    pitch_result=str(row.description),
                    pitch_result_type=str(row.type),
                    release_speed=_opt_float(row.release_speed),
                    release_spin_rate=_opt_int(row.release_spin_rate),
                    pfx_x=_opt_float(row.pfx_x),
                    pfx_z=_opt_float(row.pfx_z),
                    plate_x=_opt_float(row.plate_x),
                    plate_z=_opt_float(row.plate_z),
                    release_extension=_opt_float(row.release_extension),
                    launch_speed=_opt_float(row.launch_speed),
                    launch_angle=_opt_int(row.launch_angle),
                    hit_distance=_opt_int(row.hit_distance_sc),
                    bb_type=_opt_str(row.bb_type),
                    estimated_woba=_opt_float(row.estimated_woba_using_speedangle),
                    inning=int(row.inning),
                    is_top=is_top,
                    outs=int(row.outs_when_up),
                    balls=int(row.balls),
                    strikes=int(row.strikes),
                    runners_on_1b=pd.notna(row.on_1b),
                    runners_on_2b=pd.notna(row.on_2b),
                    runners_on_3b=pd.notna(row.on_3b),
                    bat_score=bat_score,
                    fld_score=fld_score,
                    stand=str(row.stand),
                    p_throws=str(row.p_throws),
                    pitch_number=int(row.pitch_number),
                    pa_event=_opt_str(row.events),
                    delta_run_exp=_opt_float(row.delta_run_exp),
                )
                pitches.append(pitch)

            first_row = group.iloc[0]
            sequences.append(
                GameSequence(
                    game_pk=int(game_pk),
                    game_date=str(first_row["game_date"]),
                    season=season,
                    home_team=str(first_row["home_team"]),
                    away_team=str(first_row["away_team"]),
                    perspective=perspective,
                    player_id=int(player_id_val),
                    pitches=tuple(pitches),
                )
            )

        return sequences
