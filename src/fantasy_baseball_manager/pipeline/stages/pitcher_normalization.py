from dataclasses import dataclass

from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates

_REQUIRED_RATE_KEYS = {"h", "hr", "so", "er"}


@dataclass(frozen=True)
class PitcherNormalizationConfig:
    league_babip: float = 0.300
    babip_regression_weight: float = 1.0
    lob_baseline: float = 0.73
    lob_k_sensitivity: float = 0.1
    league_k_pct: float = 0.22
    lob_regression_weight: float = 1.0
    min_lob: float = 0.65
    max_lob: float = 0.82


class PitcherNormalizationAdjuster:
    def __init__(self, config: PitcherNormalizationConfig | None = None) -> None:
        self._config = config or PitcherNormalizationConfig()

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        result: list[PlayerRates] = []
        for p in players:
            if not self._is_pitcher(p):
                result.append(p)
                continue
            if not _REQUIRED_RATE_KEYS.issubset(p.rates):
                result.append(p)
                continue
            result.append(self._adjust_pitcher(p))
        return result

    def _is_pitcher(self, p: PlayerRates) -> bool:
        return "ip_per_year" in p.metadata or "is_starter" in p.metadata

    def _adjust_pitcher(self, p: PlayerRates) -> PlayerRates:
        cfg = self._config
        rates = p.rates

        h = rates["h"]
        hr = rates["hr"]
        so = rates["so"]
        bb = rates.get("bb", 0.0)
        hbp = rates.get("hbp", 0.0)
        er = rates["er"]

        league_babip = self._derive_league_babip(p)

        # Compute observed BABIP from per-out rates: (h - hr) / (1 + h - hr - so)
        bip_denom = 1.0 + h - hr - so
        if bip_denom < 0.01:
            # Tiny BIP denominator — skip regression, pass through
            return self._build_result(p, rates, league_babip, league_babip, cfg.lob_baseline)

        observed_babip = (h - hr) / bip_denom

        # Regress BABIP toward league mean
        w = cfg.babip_regression_weight
        expected_babip = w * league_babip + (1.0 - w) * observed_babip

        # Recompute H rate from expected BABIP
        # h_new = hr + expected_babip * (bip_denom_without_h)
        # From: expected_babip = (h_new - hr) / (1 + h_new - hr - so)
        # Solving: h_new * (1 - expected_babip) = hr + expected_babip * (1 - hr - so)
        denom = 1.0 - expected_babip
        h_new = h if abs(denom) < 1e-9 else (hr + expected_babip * (1.0 - hr - so)) / denom

        # Compute K% for LOB adjustment (so is per-out, approximate k_pct)
        # k_pct ≈ so / (1 + so) converts per-out to per-PA-like
        league_k_pct = self._derive_league_k_pct(p)
        k_pct = so / (1.0 + so) if so < 1.0 else so

        # Expected LOB%
        raw_lob = cfg.lob_baseline + cfg.lob_k_sensitivity * (k_pct - league_k_pct)
        expected_lob = max(cfg.min_lob, min(cfg.max_lob, raw_lob))

        # Compute expected ER from components
        baserunners = h_new - hr + bb + hbp
        expected_er = baserunners * (1.0 - expected_lob) + hr

        # Blend expected ER with observed
        lob_w = cfg.lob_regression_weight
        er_new = lob_w * expected_er + (1.0 - lob_w) * er

        return self._build_result(p, rates, observed_babip, expected_babip, expected_lob, h_new, er_new)

    def _derive_league_babip(self, p: PlayerRates) -> float:
        avg_league = p.metadata.get("avg_league_rates")
        if isinstance(avg_league, dict):
            lg_h = avg_league.get("h")  # type: ignore[invalid-argument-type] # isinstance narrows to dict[Never, Never] in ty
            lg_hr = avg_league.get("hr")  # type: ignore[invalid-argument-type]
            lg_so = avg_league.get("so")  # type: ignore[invalid-argument-type]
            if lg_h is not None and lg_hr is not None and lg_so is not None:
                denom = 1.0 + lg_h - lg_hr - lg_so
                if denom > 0.01:
                    return (lg_h - lg_hr) / denom
        return self._config.league_babip

    def _derive_league_k_pct(self, p: PlayerRates) -> float:
        avg_league = p.metadata.get("avg_league_rates")
        if isinstance(avg_league, dict):
            lg_so = avg_league.get("so")  # type: ignore[invalid-argument-type] # isinstance narrows to dict[Never, Never] in ty
            if lg_so is not None and lg_so < 1.0:
                return lg_so / (1.0 + lg_so)
        return self._config.league_k_pct

    def _build_result(
        self,
        p: PlayerRates,
        rates: dict[str, float],
        observed_babip: float,
        expected_babip: float,
        expected_lob: float,
        h_new: float | None = None,
        er_new: float | None = None,
    ) -> PlayerRates:
        new_rates = dict(rates)
        if h_new is not None:
            new_rates["h"] = h_new
        if er_new is not None:
            new_rates["er"] = er_new

        new_metadata: PlayerMetadata = {**p.metadata}
        new_metadata["observed_babip"] = observed_babip
        new_metadata["expected_babip"] = expected_babip
        new_metadata["expected_lob_pct"] = expected_lob

        return PlayerRates(
            player_id=p.player_id,
            name=p.name,
            year=p.year,
            age=p.age,
            rates=new_rates,
            opportunities=p.opportunities,
            metadata=new_metadata,
        )
