from enum import StrEnum
from typing import NamedTuple


class PlayerType(StrEnum):
    BATTER = "batter"
    PITCHER = "pitcher"


class PlayerIdentity(NamedTuple):
    player_id: int
    player_type: PlayerType
