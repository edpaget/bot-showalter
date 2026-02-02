import logging
import re

from fantasy_baseball_manager.league.models import (
    LeagueRosters,
    PlayerMatchResult,
    TeamProjection,
)
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching."""
    name = re.sub(r"\s*\(.*?\)\s*", "", name)
    return " ".join(name.lower().split())


def match_projections(
    rosters: LeagueRosters,
    batting_projections: list[BattingProjection],
    pitching_projections: list[PitchingProjection],
    id_mapper: PlayerIdMapper,
) -> list[TeamProjection]:
    batting_by_fg: dict[str, BattingProjection] = {p.player_id: p for p in batting_projections}
    pitching_by_fg: dict[str, PitchingProjection] = {p.player_id: p for p in pitching_projections}

    # Name-indexed lookups for fallback when ID mapping is missing.
    # Duplicate names map to None to avoid ambiguous matches.
    batting_by_name: dict[str, BattingProjection | None] = {}
    for bp in batting_projections:
        key = _normalize_name(bp.name)
        batting_by_name[key] = None if key in batting_by_name else bp

    pitching_by_name: dict[str, PitchingProjection | None] = {}
    for pp in pitching_projections:
        key = _normalize_name(pp.name)
        pitching_by_name[key] = None if key in pitching_by_name else pp

    logger.debug(
        "Matching projections: %d batting, %d pitching, %d teams",
        len(batting_by_fg),
        len(pitching_by_fg),
        len(rosters.teams),
    )
    if batting_by_fg:
        sample_ids = list(batting_by_fg.keys())[:3]
        logger.debug("Sample batting projection IDs: %s", sample_ids)
    if pitching_by_fg:
        sample_ids = list(pitching_by_fg.keys())[:3]
        logger.debug("Sample pitching projection IDs: %s", sample_ids)

    team_projections: list[TeamProjection] = []

    for team in rosters.teams:
        players: list[PlayerMatchResult] = []
        unmatched_count = 0

        # Batting aggregates
        total_hr = 0.0
        total_sb = 0.0
        total_h = 0.0
        total_pa = 0.0
        total_ab = 0.0
        total_bb = 0.0
        total_hbp = 0.0
        total_r = 0.0
        total_rbi = 0.0

        # Pitching aggregates
        total_ip = 0.0
        total_so = 0.0
        total_er = 0.0
        total_p_h = 0.0
        total_p_bb = 0.0
        total_w = 0.0
        total_nsvh = 0.0

        for roster_player in team.players:
            fg_id = id_mapper.yahoo_to_fangraphs(roster_player.yahoo_id)
            batting_proj: BattingProjection | None = None
            pitching_proj: PitchingProjection | None = None
            matched = False

            if fg_id is not None:
                if roster_player.position_type == "B":
                    batting_proj = batting_by_fg.get(fg_id)
                    if batting_proj is None:
                        logger.debug(
                            "FanGraphs ID %s for %s (yahoo_id=%s) not found in %d batting projections",
                            fg_id,
                            roster_player.name,
                            roster_player.yahoo_id,
                            len(batting_by_fg),
                        )
                    if batting_proj is not None:
                        matched = True
                        total_hr += batting_proj.hr
                        total_sb += batting_proj.sb
                        total_h += batting_proj.h
                        total_pa += batting_proj.pa
                        total_ab += batting_proj.ab
                        total_bb += batting_proj.bb
                        total_hbp += batting_proj.hbp
                        total_r += batting_proj.r
                        total_rbi += batting_proj.rbi
                elif roster_player.position_type == "P":
                    pitching_proj = pitching_by_fg.get(fg_id)
                    if pitching_proj is None:
                        logger.debug(
                            "FanGraphs ID %s for %s (yahoo_id=%s) not found in %d pitching projections",
                            fg_id,
                            roster_player.name,
                            roster_player.yahoo_id,
                            len(pitching_by_fg),
                        )
                    if pitching_proj is not None:
                        matched = True
                        total_ip += pitching_proj.ip
                        total_so += pitching_proj.so
                        total_er += pitching_proj.er
                        total_p_h += pitching_proj.h
                        total_p_bb += pitching_proj.bb
                        total_w += pitching_proj.w
                        total_nsvh += pitching_proj.nsvh

            if not matched and fg_id is None:
                normalized = _normalize_name(roster_player.name)
                if roster_player.position_type == "B":
                    candidate = batting_by_name.get(normalized)
                    if candidate is not None:
                        batting_proj = candidate
                        matched = True
                        total_hr += batting_proj.hr
                        total_sb += batting_proj.sb
                        total_h += batting_proj.h
                        total_pa += batting_proj.pa
                        total_ab += batting_proj.ab
                        total_bb += batting_proj.bb
                        total_hbp += batting_proj.hbp
                        total_r += batting_proj.r
                        total_rbi += batting_proj.rbi
                        logger.debug(
                            "Name-matched %s to batting %s",
                            roster_player.name,
                            candidate.player_id,
                        )
                elif roster_player.position_type == "P":
                    candidate = pitching_by_name.get(normalized)
                    if candidate is not None:
                        pitching_proj = candidate
                        matched = True
                        total_ip += pitching_proj.ip
                        total_so += pitching_proj.so
                        total_er += pitching_proj.er
                        total_p_h += pitching_proj.h
                        total_p_bb += pitching_proj.bb
                        total_w += pitching_proj.w
                        total_nsvh += pitching_proj.nsvh
                        logger.debug(
                            "Name-matched %s to pitching %s",
                            roster_player.name,
                            candidate.player_id,
                        )

            if not matched:
                if fg_id is None:
                    logger.debug(
                        "No match for %s (yahoo_id=%s)",
                        roster_player.name,
                        roster_player.yahoo_id,
                    )
                unmatched_count += 1

            players.append(
                PlayerMatchResult(
                    roster_player=roster_player,
                    batting_projection=batting_proj,
                    pitching_projection=pitching_proj,
                    matched=matched,
                )
            )

        team_avg = total_h / total_ab if total_ab > 0 else 0.0
        team_obp = (total_h + total_bb + total_hbp) / total_pa if total_pa > 0 else 0.0
        team_era = (total_er / total_ip * 9) if total_ip > 0 else 0.0
        team_whip = (total_p_h + total_p_bb) / total_ip if total_ip > 0 else 0.0

        team_projections.append(
            TeamProjection(
                team_name=team.team_name,
                team_key=team.team_key,
                players=tuple(players),
                total_hr=total_hr,
                total_sb=total_sb,
                total_h=total_h,
                total_pa=total_pa,
                team_avg=team_avg,
                team_obp=team_obp,
                total_r=total_r,
                total_rbi=total_rbi,
                total_ip=total_ip,
                total_so=total_so,
                total_w=total_w,
                total_nsvh=total_nsvh,
                team_era=team_era,
                team_whip=team_whip,
                unmatched_count=unmatched_count,
            )
        )

    return team_projections
