"""Per-stat regression constants for stat-specific regression.

Regression amounts represent the number of plate appearances (batting)
or outs (pitching) of league-average performance added as observations
during the regression step. Lower values mean the stat stabilizes faster
and player performance is trusted more; higher values mean the stat is
noisier and regresses more toward the league mean.
"""

BATTING_REGRESSION_PA: dict[str, float] = {
    "so": 200,
    "bb": 400,
    "hr": 500,
    "hbp": 600,
    "sb": 600,
    "cs": 600,
    "singles": 800,
    "doubles": 1600,
    "triples": 1600,
    "sf": 1600,
    "sh": 1600,
    "r": 1200,
    "rbi": 1200,
}

PITCHING_REGRESSION_OUTS: dict[str, float] = {
    "so": 30,
    "bb": 60,
    "hr": 80,
    "hbp": 100,
    "h": 200,
    "er": 150,
    "w": 134,
    "sv": 134,
    "hld": 134,
    "bs": 134,
}
