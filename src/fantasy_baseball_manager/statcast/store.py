from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pandas as pd

from fantasy_baseball_manager.statcast.models import SeasonManifest

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class StatcastStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def _season_dir(self, season: int) -> Path:
        return self._data_dir / str(season)

    def _manifest_path(self, season: int) -> Path:
        return self._season_dir(season) / "manifest.json"

    def _month_path(self, season: int, month: int) -> Path:
        return self._season_dir(season) / f"statcast_{season}_{month:02d}.parquet"

    def load_manifest(self, season: int) -> SeasonManifest:
        path = self._manifest_path(season)
        if not path.exists():
            return SeasonManifest(season=season)
        data = json.loads(path.read_text())
        return SeasonManifest(
            season=data["season"],
            fetched_dates=set(data.get("fetched_dates", [])),
            total_rows=data.get("total_rows", 0),
        )

    def save_manifest(self, manifest: SeasonManifest) -> None:
        season_dir = self._season_dir(manifest.season)
        season_dir.mkdir(parents=True, exist_ok=True)
        path = self._manifest_path(manifest.season)
        data = {
            "season": manifest.season,
            "fetched_dates": sorted(manifest.fetched_dates),
            "total_rows": manifest.total_rows,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    def flush_month(self, season: int, month: int, df: pd.DataFrame) -> None:
        season_dir = self._season_dir(season)
        season_dir.mkdir(parents=True, exist_ok=True)
        path = self._month_path(season, month)
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)

    def read_month(self, season: int, month: int) -> pd.DataFrame | None:
        path = self._month_path(season, month)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def read_season(self, season: int) -> pd.DataFrame:
        season_dir = self._season_dir(season)
        if not season_dir.exists():
            return pd.DataFrame()
        parquet_files = sorted(season_dir.glob("statcast_*.parquet"))
        if not parquet_files:
            return pd.DataFrame()
        frames = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(frames, ignore_index=True)
