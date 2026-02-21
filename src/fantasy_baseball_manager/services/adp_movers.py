from datetime import date, timedelta

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.adp_movers import ADPMover, ADPMoversReport
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.protocols import ADPRepo, PlayerRepo


class ADPMoversService:
    def __init__(self, adp_repo: ADPRepo, player_repo: PlayerRepo) -> None:
        self._adp_repo = adp_repo
        self._player_repo = player_repo

    def resolve_window(self, season: int, provider: str, window_days: int = 14) -> tuple[str, str]:
        snapshots = self._adp_repo.get_snapshots(season, provider)
        if len(snapshots) < 2:
            msg = f"Need at least 2 snapshots, found {len(snapshots)}"
            raise ValueError(msg)

        current = snapshots[-1]
        target = date.fromisoformat(current) - timedelta(days=window_days)

        candidates = snapshots[:-1]
        previous = min(candidates, key=lambda s: abs(date.fromisoformat(s) - target))
        return current, previous

    def compute_adp_movers(
        self,
        season: int,
        provider: str,
        current_as_of: str,
        previous_as_of: str,
        *,
        top: int = 20,
    ) -> ADPMoversReport:
        current_records = self._adp_repo.get_by_snapshot(season, provider, current_as_of)
        previous_records = self._adp_repo.get_by_snapshot(season, provider, previous_as_of)

        all_players = self._player_repo.all()
        player_map: dict[int, Player] = {p.id: p for p in all_players if p.id is not None}

        current_best = _best_pick_per_player(current_records)
        previous_best = _best_pick_per_player(previous_records)

        current_ids = set(current_best)
        previous_ids = set(previous_best)

        matched_ids = current_ids & previous_ids
        new_ids = current_ids - previous_ids
        dropped_ids = previous_ids - current_ids

        risers: list[ADPMover] = []
        fallers: list[ADPMover] = []

        for pid in matched_ids:
            curr = current_best[pid]
            prev = previous_best[pid]
            delta = prev.rank - curr.rank
            if delta == 0:
                continue

            player = player_map.get(pid)
            name = f"{player.name_first} {player.name_last}" if player else f"Unknown ({pid})"
            direction = "riser" if delta > 0 else "faller"
            mover = ADPMover(
                player_name=name,
                position=curr.positions,
                current_rank=curr.rank,
                previous_rank=prev.rank,
                rank_delta=delta,
                direction=direction,
            )
            if delta > 0:
                risers.append(mover)
            else:
                fallers.append(mover)

        risers.sort(key=lambda m: m.rank_delta, reverse=True)
        fallers.sort(key=lambda m: m.rank_delta)

        new_entries: list[ADPMover] = []
        for pid in new_ids:
            curr = current_best[pid]
            player = player_map.get(pid)
            name = f"{player.name_first} {player.name_last}" if player else f"Unknown ({pid})"
            new_entries.append(
                ADPMover(
                    player_name=name,
                    position=curr.positions,
                    current_rank=curr.rank,
                    previous_rank=0,
                    rank_delta=0,
                    direction="new",
                )
            )
        new_entries.sort(key=lambda m: m.current_rank)

        dropped_entries: list[ADPMover] = []
        for pid in dropped_ids:
            prev = previous_best[pid]
            player = player_map.get(pid)
            name = f"{player.name_first} {player.name_last}" if player else f"Unknown ({pid})"
            dropped_entries.append(
                ADPMover(
                    player_name=name,
                    position=prev.positions,
                    current_rank=0,
                    previous_rank=prev.rank,
                    rank_delta=0,
                    direction="dropped",
                )
            )
        dropped_entries.sort(key=lambda m: m.previous_rank)

        risers = risers[:top]
        fallers = fallers[:top]

        return ADPMoversReport(
            season=season,
            provider=provider,
            current_as_of=current_as_of,
            previous_as_of=previous_as_of,
            risers=risers,
            fallers=fallers,
            new_entries=new_entries,
            dropped_entries=dropped_entries,
        )


def _best_pick_per_player(records: list[ADP]) -> dict[int, ADP]:
    best: dict[int, ADP] = {}
    for adp in records:
        existing = best.get(adp.player_id)
        if existing is None or adp.overall_pick < existing.overall_pick:
            best[adp.player_id] = adp
    return best
