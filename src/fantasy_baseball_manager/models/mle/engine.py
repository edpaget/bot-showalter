import math
from dataclasses import dataclass

from fantasy_baseball_manager.domain.league_environment import LeagueEnvironment
from fantasy_baseball_manager.domain.level_factor import LevelFactor
from fantasy_baseball_manager.domain.minor_league_batting_stats import (
    MinorLeagueBattingStats,
)
from fantasy_baseball_manager.models.mle.types import MLEConfig, TranslatedBattingLine


@dataclass(frozen=True)
class TranslatedRates:
    k_pct: float
    bb_pct: float
    iso: float
    babip: float


def compute_competition_factor(milb_rpg: float, mlb_rpg: float, level_factor: float) -> float:
    """Compute competition factor m = (milb_rpg / mlb_rpg) * level_factor."""
    return (milb_rpg / mlb_rpg) * level_factor


def translate_rates(
    *,
    k_pct: float,
    bb_pct: float,
    iso: float,
    babip: float,
    pa: int,
    k_factor: float,
    bb_factor: float,
    iso_factor: float,
    mlb_babip: float,
    config: MLEConfig,
) -> TranslatedRates:
    """Translate rate stats via level factors and BABIP regression."""
    k_pct_t = k_pct * k_factor * config.k_experience_factor
    bb_pct_t = bb_pct * bb_factor
    iso_t = iso * iso_factor

    # Bayesian BABIP shrinkage: weight = approx_bip / (approx_bip + stabilization)
    approx_bip = pa * (1.0 - k_pct - bb_pct)
    bip_weight = approx_bip / (approx_bip + config.babip_stabilization_bip)
    babip_t = bip_weight * babip + (1.0 - bip_weight) * mlb_babip

    return TranslatedRates(
        k_pct=k_pct_t,
        bb_pct=bb_pct_t,
        iso=iso_t,
        babip=babip_t,
    )


def _clamp_xbh(doubles: int, triples: int, hr: int, h: int) -> tuple[int, int]:
    """Scale 2B and 3B down proportionally if XBH exceeds H, preserving HR."""
    xbh = doubles + triples + hr
    if xbh <= h:
        return doubles, triples
    # Available slots for 2B+3B after HR
    available = max(h - hr, 0)
    total_non_hr = doubles + triples
    if total_non_hr == 0:
        return 0, 0
    ratio = available / total_non_hr
    new_doubles = int(doubles * ratio)
    new_triples = int(triples * ratio)
    # Ensure we don't exceed available due to rounding
    if new_doubles + new_triples > available:
        new_triples = available - new_doubles
    return new_doubles, new_triples


def translate_batting_line(
    stats: MinorLeagueBattingStats,
    league_env: LeagueEnvironment,
    mlb_env: LeagueEnvironment,
    level_factor: LevelFactor,
    config: MLEConfig,
) -> TranslatedBattingLine:
    """Translate a minor league batting line to an MLB-equivalent line."""
    if stats.pa < config.min_pa:
        msg = f"PA ({stats.pa}) is below minimum ({config.min_pa})"
        raise ValueError(msg)

    hbp = stats.hbp or 0
    sf = stats.sf or 0

    # 1. Compute input rates
    k_pct = stats.so / stats.pa
    bb_pct = stats.bb / stats.pa
    tb = stats.h + stats.doubles + stats.triples * 2 + stats.hr * 3
    iso = (tb / stats.ab) - (stats.h / stats.ab) if stats.ab > 0 else 0.0
    bip_denom = stats.ab - stats.so - stats.hr + sf
    babip = (stats.h - stats.hr) / bip_denom if bip_denom > 0 else 0.0

    # 2. Competition factor
    m = compute_competition_factor(
        milb_rpg=league_env.runs_per_game,
        mlb_rpg=mlb_env.runs_per_game,
        level_factor=level_factor.factor,
    )
    big_m = math.sqrt(m)

    # 3. Translate rates
    rates = translate_rates(
        k_pct=k_pct,
        bb_pct=bb_pct,
        iso=iso,
        babip=babip,
        pa=stats.pa,
        k_factor=level_factor.k_factor,
        bb_factor=level_factor.bb_factor,
        iso_factor=level_factor.iso_factor,
        mlb_babip=mlb_env.babip,
        config=config,
    )

    # 4. Reconstruct counting stats (PA preserved)
    pa = stats.pa
    so = round(rates.k_pct * pa)
    bb = round(rates.bb_pct * pa)
    ab = pa - bb - hbp - sf
    hr = round(stats.hr * m)
    bip = ab - so - hr + sf
    bip = max(bip, 0)
    h_from_bip = round(rates.babip * bip) if bip > 0 else 0
    h = h_from_bip + hr

    # Scale extra-base hits
    doubles = round(stats.doubles * big_m)
    triples = round(stats.triples * m * 0.85)

    # Clamp XBH if they exceed H
    doubles, triples = _clamp_xbh(doubles, triples, hr, h)

    # Ensure h doesn't exceed ab
    h = min(h, ab)

    # 5. Derive rate stats
    avg = h / ab if ab > 0 else 0.0
    obp_denom = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denom if obp_denom > 0 else 0.0
    tb_t = h + doubles + triples * 2 + hr * 3
    slg = tb_t / ab if ab > 0 else 0.0
    derived_iso = slg - avg
    k_pct_out = so / pa if pa > 0 else 0.0
    bb_pct_out = bb / pa if pa > 0 else 0.0
    bip_out = ab - so - hr + sf
    babip_out = (h - hr) / bip_out if bip_out > 0 else 0.0

    return TranslatedBattingLine(
        player_id=stats.player_id,
        season=stats.season,
        source_level=stats.level,
        pa=pa,
        ab=ab,
        h=h,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
        avg=avg,
        obp=obp,
        slg=slg,
        k_pct=k_pct_out,
        bb_pct=bb_pct_out,
        iso=derived_iso,
        babip=babip_out,
    )
