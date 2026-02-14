from dataclasses import dataclass


@dataclass(frozen=True)
class Player:
    name_first: str
    name_last: str
    id: int | None = None
    mlbam_id: int | None = None
    fangraphs_id: int | None = None
    bbref_id: str | None = None
    retro_id: str | None = None
    bats: str | None = None
    throws: str | None = None
    birth_date: str | None = None
    position: str | None = None


@dataclass(frozen=True)
class Team:
    abbreviation: str
    name: str
    league: str
    division: str
    id: int | None = None
