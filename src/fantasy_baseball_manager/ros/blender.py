from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
    PITCHING_REGRESSION_OUTS,
)

BATTING_COMPONENT_STATS: tuple[str, ...] = (
    "singles", "doubles", "triples", "hr", "bb", "so", "hbp", "sf", "sh", "sb", "cs", "r", "rbi",
)

PITCHING_COMPONENT_STATS: tuple[str, ...] = (
    "so", "bb", "hr", "hbp", "h", "er", "w", "sv", "hld", "bs",
)


class BayesianBlender:
    def __init__(
        self,
        batting_regression: dict[str, float] | None = None,
        pitching_regression: dict[str, float] | None = None,
    ) -> None:
        self._batting_reg = batting_regression or dict(BATTING_REGRESSION_PA)
        self._pitching_reg = pitching_regression or dict(PITCHING_REGRESSION_OUTS)

    def blend_batting(
        self,
        preseason: BattingProjection,
        actuals: BattingSeasonStats,
    ) -> BattingProjection:
        pre_pa = preseason.pa
        act_pa = float(actuals.pa)
        remaining_pa = max(0.0, pre_pa - act_pa)

        blended: dict[str, float] = {}
        for stat in BATTING_COMPONENT_STATS:
            pre_val = getattr(preseason, stat)
            act_val = float(getattr(actuals, stat))
            reg = self._batting_reg.get(stat, 1200.0)

            pre_rate = pre_val / pre_pa if pre_pa > 0 else 0.0
            act_rate = act_val / act_pa if act_pa > 0 else 0.0

            blended_rate = (pre_rate * reg + act_rate * act_pa) / (reg + act_pa) if (reg + act_pa) > 0 else 0.0
            blended[stat] = blended_rate * remaining_pa

        h = blended["singles"] + blended["doubles"] + blended["triples"] + blended["hr"]
        ab = remaining_pa - blended["bb"] - blended["hbp"] - blended["sf"] - blended["sh"]

        return BattingProjection(
            player_id=preseason.player_id,
            name=preseason.name,
            year=preseason.year,
            age=preseason.age,
            pa=remaining_pa,
            ab=ab,
            h=h,
            singles=blended["singles"],
            doubles=blended["doubles"],
            triples=blended["triples"],
            hr=blended["hr"],
            bb=blended["bb"],
            so=blended["so"],
            hbp=blended["hbp"],
            sf=blended["sf"],
            sh=blended["sh"],
            sb=blended["sb"],
            cs=blended["cs"],
            r=blended["r"],
            rbi=blended["rbi"],
        )

    def blend_pitching(
        self,
        preseason: PitchingProjection,
        actuals: PitchingSeasonStats,
    ) -> PitchingProjection:
        pre_outs = preseason.ip * 3
        act_outs = actuals.ip * 3
        remaining_ip = max(0.0, preseason.ip - actuals.ip)
        remaining_outs = remaining_ip * 3

        blended: dict[str, float] = {}
        for stat in PITCHING_COMPONENT_STATS:
            # preseason projection stores nsvh, not individual sv/hld/bs
            pre_val = 0.0 if stat in ("sv", "hld", "bs") else getattr(preseason, stat)
            act_val = float(getattr(actuals, stat))
            reg = self._pitching_reg.get(stat, 134.0)

            pre_rate = pre_val / pre_outs if pre_outs > 0 else 0.0
            act_rate = act_val / act_outs if act_outs > 0 else 0.0

            blended_rate = (pre_rate * reg + act_rate * act_outs) / (reg + act_outs) if (reg + act_outs) > 0 else 0.0
            blended[stat] = blended_rate * remaining_outs

        era = (blended["er"] / remaining_ip) * 9 if remaining_ip > 0 else 0.0
        whip = (blended["h"] + blended["bb"]) / remaining_ip if remaining_ip > 0 else 0.0
        nsvh = blended["sv"] + blended["hld"] - blended["bs"]

        is_starter = (actuals.gs / actuals.g) >= 0.5 if actuals.g > 0 else (preseason.gs > 0)
        if is_starter:
            proj_gs = remaining_ip / 6.0
            proj_g = proj_gs
        else:
            proj_gs = 0.0
            proj_g = remaining_ip

        return PitchingProjection(
            player_id=preseason.player_id,
            name=preseason.name,
            year=preseason.year,
            age=preseason.age,
            ip=remaining_ip,
            g=proj_g,
            gs=proj_gs,
            er=blended["er"],
            h=blended["h"],
            bb=blended["bb"],
            so=blended["so"],
            hr=blended["hr"],
            hbp=blended["hbp"],
            era=era,
            whip=whip,
            w=blended["w"],
            nsvh=nsvh,
        )
