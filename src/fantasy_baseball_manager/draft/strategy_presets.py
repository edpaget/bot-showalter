from fantasy_baseball_manager.draft.rules import (
    MaxPositionCount,
    PitcherBatterRatio,
    PositionNotBeforeRound,
)
from fantasy_baseball_manager.draft.simulation_models import DraftStrategy
from fantasy_baseball_manager.valuation.models import StatCategory

STRATEGY_PRESETS: dict[str, DraftStrategy] = {
    "balanced": DraftStrategy(
        name="balanced",
        category_weights={},
        rules=(
            MaxPositionCount(position="C", max_count=2),
            PositionNotBeforeRound(position="RP", earliest_round=10),
        ),
    ),
    "power_hitting": DraftStrategy(
        name="power_hitting",
        category_weights={
            StatCategory.HR: 1.3,
            StatCategory.RBI: 1.2,
            StatCategory.R: 1.1,
        },
        rules=(
            MaxPositionCount(position="C", max_count=2),
            PitcherBatterRatio(max_pitcher_fraction=0.45),
        ),
    ),
    "speed": DraftStrategy(
        name="speed",
        category_weights={
            StatCategory.SB: 1.4,
            StatCategory.R: 1.2,
        },
        rules=(
            MaxPositionCount(position="C", max_count=1),
            PitcherBatterRatio(max_pitcher_fraction=0.45),
            PositionNotBeforeRound(position="RP", earliest_round=10),
        ),
    ),
    "pitching_heavy": DraftStrategy(
        name="pitching_heavy",
        category_weights={
            StatCategory.K: 1.25,
            StatCategory.ERA: 1.15,
            StatCategory.WHIP: 1.15,
            StatCategory.W: 1.1,
        },
        rules=(
            MaxPositionCount(position="C", max_count=2),
            PitcherBatterRatio(max_pitcher_fraction=0.5),
        ),
    ),
    "punt_saves": DraftStrategy(
        name="punt_saves",
        category_weights={
            StatCategory.NSVH: 0.0,
        },
        rules=(
            MaxPositionCount(position="C", max_count=2),
            PositionNotBeforeRound(position="RP", earliest_round=20),
        ),
    ),
}
