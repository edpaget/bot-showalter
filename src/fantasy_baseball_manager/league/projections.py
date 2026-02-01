from fantasy_baseball_manager.league.models import (
    LeagueRosters,
    PlayerMatchResult,
    TeamProjection,
)
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper


def match_projections(
    rosters: LeagueRosters,
    batting_projections: list[BattingProjection],
    pitching_projections: list[PitchingProjection],
    id_mapper: PlayerIdMapper,
) -> list[TeamProjection]:
    batting_by_fg: dict[str, BattingProjection] = {p.player_id: p for p in batting_projections}
    pitching_by_fg: dict[str, PitchingProjection] = {p.player_id: p for p in pitching_projections}

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

        # Pitching aggregates
        total_ip = 0.0
        total_so = 0.0
        total_er = 0.0
        total_p_h = 0.0
        total_p_bb = 0.0

        for roster_player in team.players:
            fg_id = id_mapper.yahoo_to_fangraphs(roster_player.yahoo_id)
            batting_proj: BattingProjection | None = None
            pitching_proj: PitchingProjection | None = None
            matched = False

            if fg_id is not None:
                if roster_player.position_type == "B":
                    batting_proj = batting_by_fg.get(fg_id)
                    if batting_proj is not None:
                        matched = True
                        total_hr += batting_proj.hr
                        total_sb += batting_proj.sb
                        total_h += batting_proj.h
                        total_pa += batting_proj.pa
                        total_ab += batting_proj.ab
                        total_bb += batting_proj.bb
                        total_hbp += batting_proj.hbp
                elif roster_player.position_type == "P":
                    pitching_proj = pitching_by_fg.get(fg_id)
                    if pitching_proj is not None:
                        matched = True
                        total_ip += pitching_proj.ip
                        total_so += pitching_proj.so
                        total_er += pitching_proj.er
                        total_p_h += pitching_proj.h
                        total_p_bb += pitching_proj.bb

            if not matched:
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
                total_ip=total_ip,
                total_so=total_so,
                team_era=team_era,
                team_whip=team_whip,
                unmatched_count=unmatched_count,
            )
        )

    return team_projections
