from dataclasses import dataclass


@dataclass(frozen=True)
class SeasonLine:
    stats: dict[str, float]
    pa: int = 0
    ip: float = 0.0
    g: int = 0
    gs: int = 0


@dataclass(frozen=True)
class LeagueAverages:
    rates: dict[str, float]


@dataclass(frozen=True)
class MarcelInput:
    weighted_rates: dict[str, float]
    weighted_pt: float
    league_rates: dict[str, float]
    age: int
    seasons: tuple[SeasonLine, ...]


@dataclass(frozen=True)
class MarcelProjection:
    player_id: int
    projected_season: int
    age: int
    stats: dict[str, float]
    rates: dict[str, float]
    pa: int = 0
    ip: float = 0.0


@dataclass(frozen=True)
class MarcelConfig:
    batting_weights: tuple[float, ...] = (5.0, 4.0, 3.0)
    pitching_weights: tuple[float, ...] = (3.0, 2.0, 1.0)
    batting_regression_pa: float = 1200.0
    pitching_regression_ip: float = 134.0
    batting_baseline_pa: float = 200.0
    pitching_starter_baseline_ip: float = 60.0
    pitching_reliever_baseline_ip: float = 25.0
    pa_weights: tuple[float, ...] = (0.5, 0.1)
    ip_weights: tuple[float, ...] = (0.5, 0.1)
    age_peak: int = 29
    age_improvement_rate: float = 0.006
    age_decline_rate: float = 0.003
    reliever_gs_ratio: float = 0.5
    batting_categories: tuple[str, ...] = (
        "h",
        "doubles",
        "triples",
        "hr",
        "r",
        "rbi",
        "bb",
        "so",
        "sb",
        "cs",
        "hbp",
        "sf",
    )
    pitching_categories: tuple[str, ...] = (
        "w",
        "l",
        "sv",
        "h",
        "er",
        "hr",
        "bb",
        "so",
    )
