from dataclasses import dataclass


@dataclass(frozen=True)
class YahooPlayerMap:
    yahoo_player_key: str
    player_id: int
    player_type: str
    yahoo_name: str
    yahoo_team: str
    yahoo_positions: str
    id: int | None = None
