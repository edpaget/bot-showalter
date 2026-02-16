import math
from collections.abc import Sequence
from dataclasses import dataclass

from fantasy_baseball_manager.domain.league_environment import LeagueEnvironment
from fantasy_baseball_manager.domain.level_factor import LevelFactor
from fantasy_baseball_manager.domain.minor_league_batting_stats import (
    MinorLeagueBattingStats,
)
from fantasy_baseball_manager.models.mle.age_adjustment import compute_age_adjustment
from fantasy_baseball_manager.models.mle.types import (
    AgeAdjustmentConfig,
    MLEConfig,
    TranslatedBattingLine,
)


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
    age_config: AgeAdjustmentConfig | None = None,
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

    # 3b. Apply age adjustment to translated rates
    if age_config is not None:
        full_adj = compute_age_adjustment(age=stats.age, level=stats.level, config=age_config)
        dampened_adj = 1.0 + (full_adj - 1.0) * 0.5
        rates = TranslatedRates(
            k_pct=rates.k_pct * (1.0 / dampened_adj),
            bb_pct=rates.bb_pct * dampened_adj,
            iso=rates.iso * full_adj,
            babip=rates.babip * full_adj,
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


def _derive_rates(
    *,
    pa: int,
    ab: int,
    h: int,
    doubles: int,
    triples: int,
    hr: int,
    bb: int,
    so: int,
    hbp: int,
    sf: int,
) -> dict[str, float]:
    """Derive rate stats from counting stats."""
    avg = h / ab if ab > 0 else 0.0
    obp_denom = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denom if obp_denom > 0 else 0.0
    tb = h + doubles + triples * 2 + hr * 3
    slg = tb / ab if ab > 0 else 0.0
    iso = slg - avg
    k_pct = so / pa if pa > 0 else 0.0
    bb_pct = bb / pa if pa > 0 else 0.0
    bip = ab - so - hr + sf
    babip = (h - hr) / bip if bip > 0 else 0.0
    return {
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "iso": iso,
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "babip": babip,
    }


def combine_translated_lines(lines: Sequence[TranslatedBattingLine]) -> TranslatedBattingLine:
    """PA-weight rate stats and sum counting stats across multiple translated lines."""
    if len(lines) == 1:
        return lines[0]

    total_pa = sum(line.pa for line in lines)
    total_ab = sum(line.ab for line in lines)
    total_h = sum(line.h for line in lines)
    total_doubles = sum(line.doubles for line in lines)
    total_triples = sum(line.triples for line in lines)
    total_hr = sum(line.hr for line in lines)
    total_bb = sum(line.bb for line in lines)
    total_so = sum(line.so for line in lines)
    total_hbp = sum(line.hbp for line in lines)
    total_sf = sum(line.sf for line in lines)

    rates = _derive_rates(
        pa=total_pa,
        ab=total_ab,
        h=total_h,
        doubles=total_doubles,
        triples=total_triples,
        hr=total_hr,
        bb=total_bb,
        so=total_so,
        hbp=total_hbp,
        sf=total_sf,
    )

    # PA-weight the rate stats that can't be derived from counting stats alone
    k_pct = sum(line.k_pct * line.pa for line in lines) / total_pa
    bb_pct = sum(line.bb_pct * line.pa for line in lines) / total_pa
    iso = sum(line.iso * line.pa for line in lines) / total_pa
    babip = sum(line.babip * line.pa for line in lines) / total_pa

    return TranslatedBattingLine(
        player_id=lines[0].player_id,
        season=lines[0].season,
        source_level=lines[0].source_level,
        pa=total_pa,
        ab=total_ab,
        h=total_h,
        doubles=total_doubles,
        triples=total_triples,
        hr=total_hr,
        bb=total_bb,
        so=total_so,
        hbp=total_hbp,
        sf=total_sf,
        avg=rates["avg"],
        obp=rates["obp"],
        slg=rates["slg"],
        k_pct=k_pct,
        bb_pct=bb_pct,
        iso=iso,
        babip=babip,
    )


def apply_recency_weights(
    season_lines: Sequence[tuple[TranslatedBattingLine, float]],
) -> TranslatedBattingLine:
    """Apply Marcel-style recency weights across seasons.

    Each tuple is (translated_line, weight). Produces a single composite line
    with rates weighted by pa * recency_weight.
    """
    if len(season_lines) == 1:
        return season_lines[0][0]

    # Filter out zero-weight seasons
    active = [(line, w) for line, w in season_lines if w > 0]
    if len(active) == 1:
        return active[0][0]

    weighted_pa = sum(line.pa * w for line, w in active)

    # PA-weight rate stats by recency
    k_pct = sum(line.k_pct * line.pa * w for line, w in active) / weighted_pa
    bb_pct = sum(line.bb_pct * line.pa * w for line, w in active) / weighted_pa
    iso = sum(line.iso * line.pa * w for line, w in active) / weighted_pa
    babip = sum(line.babip * line.pa * w for line, w in active) / weighted_pa

    # Weighted counting stats
    total_pa = round(weighted_pa)
    total_ab = round(sum(line.ab * w for line, w in active))
    total_h = round(sum(line.h * w for line, w in active))
    total_doubles = round(sum(line.doubles * w for line, w in active))
    total_triples = round(sum(line.triples * w for line, w in active))
    total_hr = round(sum(line.hr * w for line, w in active))
    total_bb = round(sum(line.bb * w for line, w in active))
    total_so = round(sum(line.so * w for line, w in active))
    total_hbp = round(sum(line.hbp * w for line, w in active))
    total_sf = round(sum(line.sf * w for line, w in active))

    rates = _derive_rates(
        pa=total_pa,
        ab=total_ab,
        h=total_h,
        doubles=total_doubles,
        triples=total_triples,
        hr=total_hr,
        bb=total_bb,
        so=total_so,
        hbp=total_hbp,
        sf=total_sf,
    )

    return TranslatedBattingLine(
        player_id=active[0][0].player_id,
        season=active[0][0].season,
        source_level=active[0][0].source_level,
        pa=total_pa,
        ab=total_ab,
        h=total_h,
        doubles=total_doubles,
        triples=total_triples,
        hr=total_hr,
        bb=total_bb,
        so=total_so,
        hbp=total_hbp,
        sf=total_sf,
        avg=rates["avg"],
        obp=rates["obp"],
        slg=rates["slg"],
        k_pct=k_pct,
        bb_pct=bb_pct,
        iso=iso,
        babip=babip,
    )


def regress_to_mlb(
    rates: dict[str, float],
    mlb_rates: dict[str, float],
    effective_pa: float,
    regression_pa: float,
) -> dict[str, float]:
    """Regress rates toward MLB league average.

    Formula: (rate * effective_pa + mlb_rate * regression_pa) / (effective_pa + regression_pa)
    """
    denominator = effective_pa + regression_pa
    return {stat: (rate * effective_pa + mlb_rates[stat] * regression_pa) / denominator for stat, rate in rates.items()}
