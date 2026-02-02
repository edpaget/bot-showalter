from dataclasses import dataclass

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection


@dataclass(frozen=True)
class RosterPlayer:
    yahoo_id: str
    name: str
    position_type: str
    eligible_positions: tuple[str, ...]


@dataclass(frozen=True)
class TeamRoster:
    team_key: str
    team_name: str
    players: tuple[RosterPlayer, ...]


@dataclass(frozen=True)
class LeagueRosters:
    league_key: str
    teams: tuple[TeamRoster, ...]


@dataclass(frozen=True)
class PlayerMatchResult:
    roster_player: RosterPlayer
    batting_projection: BattingProjection | None
    pitching_projection: PitchingProjection | None
    matched: bool


@dataclass(frozen=True)
class TeamProjection:
    team_name: str
    team_key: str
    players: tuple[PlayerMatchResult, ...]
    total_hr: float
    total_sb: float
    total_h: float
    total_pa: float
    team_avg: float
    team_obp: float
    total_r: float
    total_rbi: float
    total_ip: float
    total_so: float
    total_w: float
    total_nsvh: float
    team_era: float
    team_whip: float
    unmatched_count: int
