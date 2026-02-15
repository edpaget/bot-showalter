from fantasy_baseball_manager.domain.projection import Projection


def pt_projection_to_domain(
    player_id: int,
    projected_season: int,
    pt: float,
    pitcher: bool,
    version: str,
) -> Projection:
    """Convert a playing-time projection to a domain Projection."""
    if pitcher:
        stat_json: dict[str, object] = {"ip": pt}
        player_type = "pitcher"
    else:
        stat_json = {"pa": round(pt)}
        player_type = "batter"

    return Projection(
        player_id=player_id,
        season=projected_season,
        system="playing_time",
        version=version,
        player_type=player_type,
        stat_json=stat_json,
    )
