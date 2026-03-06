from enum import StrEnum


class Position(StrEnum):
    C = "C"
    FIRST_BASE = "1B"
    SECOND_BASE = "2B"
    THIRD_BASE = "3B"
    SS = "SS"
    LF = "LF"
    CF = "CF"
    RF = "RF"
    OF = "OF"
    DH = "DH"
    UTIL = "UTIL"
    SP = "SP"
    RP = "RP"
    P = "P"


_RAW_LOOKUP: dict[str, Position] = {m.value: m for m in Position}


def position_from_raw(s: str) -> Position:
    upper = s.strip().upper()
    result = _RAW_LOOKUP.get(upper)
    if result is not None:
        return result
    msg = f"Unknown position: {s!r}"
    raise ValueError(msg)


OF_POSITIONS: frozenset[Position] = frozenset({Position.LF, Position.CF, Position.RF, Position.OF})


def consolidate_outfield(pos: Position) -> Position:
    if pos in OF_POSITIONS:
        return Position.OF
    return pos
