from fantasy_baseball_manager.features import batting, player
from fantasy_baseball_manager.features.transforms import (
    BATTED_BALL,
    EXPECTED_STATS,
    PITCH_MIX,
    PLATE_DISCIPLINE,
    SPIN_PROFILE,
)
from fantasy_baseball_manager.features.types import Feature, TransformFeature

_COUNTING_STATS = ("pa", "ab", "h", "hr", "rbi", "r", "sb", "cs", "bb", "so")

STANDARD_BATTING_COUNTING: tuple[Feature, ...] = tuple(
    batting.col(stat).lag(lag).alias(f"{stat}_{lag}") for stat in _COUNTING_STATS for lag in (1, 2, 3)
)

STANDARD_BATTING_RATES: tuple[Feature, ...] = (
    batting.col("avg").lag(1).alias("avg_1"),
    batting.col("obp").lag(1).alias("obp_1"),
    batting.col("slg").lag(1).alias("slg_1"),
    batting.col("ops").lag(1).alias("ops_1"),
    batting.col("woba").lag(1).alias("woba_1"),
    batting.col("wrc_plus").lag(1).alias("wrc_plus_1"),
)

PLAYER_METADATA: tuple[Feature, ...] = (player.age(),)

STATCAST_PITCH_MIX: tuple[TransformFeature, ...] = (PITCH_MIX,)

STATCAST_BATTED_BALL: tuple[TransformFeature, ...] = (BATTED_BALL,)

STATCAST_PLATE_DISCIPLINE: tuple[TransformFeature, ...] = (PLATE_DISCIPLINE,)

STATCAST_EXPECTED_STATS: tuple[TransformFeature, ...] = (EXPECTED_STATS,)

STATCAST_SPIN_PROFILE: tuple[TransformFeature, ...] = (SPIN_PROFILE,)
