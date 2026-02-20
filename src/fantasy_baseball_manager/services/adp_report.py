import logging

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.adp_report import ValueOverADP, ValueOverADPReport
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos.protocols import ADPRepo, PlayerRepo, ValuationRepo

logger = logging.getLogger(__name__)

_PITCHER_POSITIONS = {"SP", "RP"}


def _is_pitcher_adp(adp: ADP) -> bool:
    return all(p.strip() in _PITCHER_POSITIONS for p in adp.positions.split(",") if p.strip())


class ADPReportService:
    def __init__(self, player_repo: PlayerRepo, valuation_repo: ValuationRepo, adp_repo: ADPRepo) -> None:
        self._player_repo = player_repo
        self._valuation_repo = valuation_repo
        self._adp_repo = adp_repo

    def compute_value_over_adp(
        self,
        season: int,
        system: str,
        version: str,
        provider: str = "fantasypros",
        player_type: str | None = None,
        position: str | None = None,
        top: int | None = None,
    ) -> ValueOverADPReport:
        # 1. Fetch valuations, filter by version
        valuations = self._valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        # 2. Apply player_type and position filters
        if player_type is not None:
            valuations = [v for v in valuations if v.player_type == player_type]
        if position is not None:
            valuations = [v for v in valuations if v.position == position]

        # 3. Fetch ADP
        adp_list = self._adp_repo.get_by_season(season, provider=provider)

        # 4. Build player name map
        all_players = self._player_repo.all()
        player_map: dict[int, Player] = {p.id: p for p in all_players if p.id is not None}

        # 5. Build ADP lookup: dict[int, ADP] keyed by player_id
        adp_by_player: dict[int, list[ADP]] = {}
        for adp in adp_list:
            adp_by_player.setdefault(adp.player_id, []).append(adp)

        adp_lookup: dict[int, ADP] = {}
        for pid, entries in adp_by_player.items():
            if player_type is not None:
                # Try to find an entry matching the player type
                is_pitcher_filter = player_type == "pitcher"
                matching = [e for e in entries if _is_pitcher_adp(e) == is_pitcher_filter]
                if matching:
                    adp_lookup[pid] = min(matching, key=lambda a: a.overall_pick)
                else:
                    # Fallback to lowest overall_pick
                    adp_lookup[pid] = min(entries, key=lambda a: a.overall_pick)
            else:
                # No filter: use lowest overall_pick (most valuable)
                adp_lookup[pid] = min(entries, key=lambda a: a.overall_pick)

        # 6. Build valuation lookup
        val_lookup = {v.player_id: v for v in valuations}

        # 7. Join
        matched: list[ValueOverADP] = []
        unranked: list[ValueOverADP] = []

        for pid, val in val_lookup.items():
            player = player_map.get(pid)
            player_name = f"{player.name_first} {player.name_last}" if player else f"Unknown ({pid})"

            adp_entry = adp_lookup.get(pid)
            if adp_entry is not None:
                adp_rank = adp_entry.rank
                rank_delta = adp_rank - val.rank
                matched.append(
                    ValueOverADP(
                        player_id=pid,
                        player_name=player_name,
                        player_type=val.player_type,
                        position=val.position,
                        adp_positions=adp_entry.positions,
                        zar_rank=val.rank,
                        zar_value=val.value,
                        adp_rank=adp_rank,
                        adp_pick=adp_entry.overall_pick,
                        rank_delta=rank_delta,
                        provider=provider,
                    )
                )
            elif val.rank <= 300:
                unranked.append(
                    ValueOverADP(
                        player_id=pid,
                        player_name=player_name,
                        player_type=val.player_type,
                        position=val.position,
                        adp_positions="",
                        zar_rank=val.rank,
                        zar_value=val.value,
                        adp_rank=0,
                        adp_pick=0.0,
                        rank_delta=0,
                        provider=provider,
                    )
                )

        # 8. Sort into buy_targets and avoid_list
        buy_targets = [m for m in matched if m.rank_delta > 0]
        avoid_list = [m for m in matched if m.rank_delta < 0]

        buy_targets.sort(key=lambda v: v.rank_delta, reverse=True)
        avoid_list.sort(key=lambda v: v.rank_delta)
        unranked.sort(key=lambda v: v.zar_rank)

        # 9. Apply top limit
        if top is not None:
            buy_targets = buy_targets[:top]
            avoid_list = avoid_list[:top]
            unranked = unranked[:top]

        return ValueOverADPReport(
            season=season,
            system=system,
            version=version,
            provider=provider,
            buy_targets=buy_targets,
            avoid_list=avoid_list,
            unranked_valuable=unranked,
            n_matched=len(matched),
        )
