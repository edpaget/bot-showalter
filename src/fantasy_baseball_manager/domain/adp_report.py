from dataclasses import dataclass


@dataclass(frozen=True)
class ValueOverADP:
    player_id: int
    player_name: str
    player_type: str
    position: str
    adp_positions: str
    zar_rank: int
    zar_value: float
    adp_rank: int
    adp_pick: float
    rank_delta: int
    provider: str


@dataclass(frozen=True)
class ValueOverADPReport:
    season: int
    system: str
    version: str
    provider: str
    buy_targets: list[ValueOverADP]
    avoid_list: list[ValueOverADP]
    unranked_valuable: list[ValueOverADP]
    n_matched: int
