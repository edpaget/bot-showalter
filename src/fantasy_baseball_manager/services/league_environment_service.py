from fantasy_baseball_manager.domain.league_environment import LeagueEnvironment
from fantasy_baseball_manager.repos.protocols import LeagueEnvironmentRepo, MinorLeagueBattingStatsRepo


class LeagueEnvironmentService:
    def __init__(self, stats_repo: MinorLeagueBattingStatsRepo, env_repo: LeagueEnvironmentRepo) -> None:
        self._stats_repo = stats_repo
        self._env_repo = env_repo

    def compute_for_league(self, league: str, season: int, level: str) -> LeagueEnvironment:
        all_stats = self._stats_repo.get_by_season_level(season, level)
        stats = [s for s in all_stats if s.league == league]
        if not stats:
            raise ValueError(f"No stats found for league={league!r}, season={season}, level={level!r}")

        total_g = sum(s.g for s in stats)
        total_pa = sum(s.pa for s in stats)
        total_ab = sum(s.ab for s in stats)
        total_h = sum(s.h for s in stats)
        total_2b = sum(s.doubles for s in stats)
        total_3b = sum(s.triples for s in stats)
        total_hr = sum(s.hr for s in stats)
        total_r = sum(s.r for s in stats)
        total_bb = sum(s.bb for s in stats)
        total_so = sum(s.so for s in stats)
        total_hbp = sum(s.hbp or 0 for s in stats)
        total_sf = sum(s.sf or 0 for s in stats)

        avg = total_h / total_ab
        obp = (total_h + total_bb + total_hbp) / (total_ab + total_bb + total_hbp + total_sf)
        total_bases = total_h + total_2b + 2 * total_3b + 3 * total_hr
        slg = total_bases / total_ab
        k_pct = total_so / total_pa
        bb_pct = total_bb / total_pa
        hr_per_pa = total_hr / total_pa
        babip_denom = total_ab - total_so - total_hr + total_sf
        babip = (total_h - total_hr) / babip_denom
        runs_per_game = total_r / total_g

        return LeagueEnvironment(
            league=league,
            season=season,
            level=level,
            runs_per_game=runs_per_game,
            avg=avg,
            obp=obp,
            slg=slg,
            k_pct=k_pct,
            bb_pct=bb_pct,
            hr_per_pa=hr_per_pa,
            babip=babip,
        )

    def compute_and_persist(self, league: str, season: int, level: str) -> LeagueEnvironment:
        env = self.compute_for_league(league, season, level)
        self._env_repo.upsert(env)
        return env

    def compute_for_season_level(self, season: int, level: str) -> int:
        all_stats = self._stats_repo.get_by_season_level(season, level)
        leagues = {s.league for s in all_stats}
        count = 0
        for league in sorted(leagues):
            self.compute_and_persist(league, season, level)
            count += 1
        return count
