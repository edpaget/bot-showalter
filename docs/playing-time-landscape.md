# Modeling MLB Playing Time: State of the Art

## Why It's Hard

Playing time is widely recognized as the hardest element to project in baseball. Even PECOTA's [2026 update](https://www.baseballprospectus.com/news/article/104636/pecota-2026-updates-and-ongoing-challenges/) identifies playing time unpredictability as a fundamental, ongoing challenge. FanGraphs' own [2024 projection review](https://fantasy.fangraphs.com/2024-projection-review-batter-playing-time/) found that **every projection system systematically overprojects elite hitters' playing time** — overprojecting injury-prone stars like Mike Trout and Kris Bryant by large margins.

The problem is that playing time for established players is dominated by stochastic events — injuries, trades, role changes, managerial decisions — that no amount of historical feature engineering can predict. A healthy full-time regular records anywhere from 600 to 750 PA in a season; catchers typically 450–600. The range is wide and the causes of variance are largely exogenous.

## How the Major Systems Handle It

### Marcel (the baseline)

Tom Tango's [Marcel](https://www.baseball-reference.com/about/marcels.shtml) is the "minimum level of competence" baseline. Its playing time formula is:

```
PA = 0.5 * PA_year1 + 0.1 * PA_year2 + 200
```

Purely mechanical — no injury, age, or roster context. The constant of 200 serves as regression toward the mean. Despite its simplicity, Marcel has proven [very difficult to beat](https://www.beyondtheboxscore.com/2016/2/22/11079186/projections-marcel-pecota-zips-steamer-explained-guide-math-is-fun) for top-of-roster players.

### Steamer / ZiPS / THE BAT

These systems project **rates** (per-PA performance) separately from **playing time**. Playing time is a secondary concern, sometimes hand-tuned.

- **Steamer** uses weighted average of past performance regressed toward league average, similar to Marcel but with varying weights and regression levels across stats.
- **ZiPS** uses weighted averages of the previous four seasons plus a PECOTA-like comparable-player system for age adjustment, with a database extending to the early 1970s.
- **[THE BAT](https://fantasy.fangraphs.com/an-intro-to-the-bat/)** layers in park factors, platoon splits, air density, umpire tendencies, and schedule-aware adjustments on top of standard weighted averages. THE BAT X adds Statcast data for hitters. Consistently ranks as the [best original projection system](https://www.fantasypros.com/2026/02/most-accurate-fantasy-baseball-projections-2025-results/).

All of these systems deliberately **overproject playing time by design** — this gives users more information about player ability rather than trying to predict team usage decisions.

### FanGraphs Depth Charts / RosterResource

The current industry standard is a **hybrid human + model approach**. FanGraphs [Depth Charts](https://blogs.fangraphs.com/introducing-rosterresource-depth-charts/) blend Steamer and ZiPS rate projections 50/50, then apply playing time allocations from a human editor (Jason Martinez at RosterResource) who maintains depth charts based on roster moves, spring training, and transactions — updated daily.

This separation of concerns — models for rates, human judgment for playing time — is what most consumers of baseball projections actually use.

### PECOTA

PECOTA is unique in producing a [full probability distribution](https://en.wikipedia.org/wiki/PECOTA) of outcomes via percentile forecasts (10th/25th/50th/75th/90th). It uses comparable-player matching based on performance, age, and body type. The distributional output implicitly captures playing time variance — a player with high injury risk will have a wider spread between P10 and P90. Playing time for some players is determined by Depth Charts; others get hypothetical values representing what they'd do if given a full-time role.

## Empirical Results: What Actually Works

### The Hardball Times Regression Model

A [Hardball Times study](https://tht.fangraphs.com/projecting-playing-time/) using multiple linear regression found that five variables dramatically outperform Marcel:

| Factor | Why It Matters |
|--------|---------------|
| Prior year PA | Most recent year only (not 2–3 year weighted average) |
| Prior year WAR | Better players get more opportunities |
| Prior year DL days | Injury history predicts future missed time |
| Age | Older players miss more time |
| Starter status | Opening Day starter vs. bench |

This model achieved **R² = 0.74** vs. Marcel's R² ≈ 0.08 — explaining nearly 10x more variance. The starter status variable alone accounts for much of the gap; removing it still yields R² = 0.67.

### Projection Aggregates

The [2024 FanGraphs playing time review](https://fantasy.fangraphs.com/2024-projection-review-batter-playing-time/) found that **blending a computer-generated adjustment model with projection aggregates** was optimal:

- Simple average of all projections: RMSE 145.7
- Average of four free projections: RMSE 153.2
- Marcel alone: RMSE 142.2
- Optimal blend (model + aggregate): ~50/50 weighting between model-based adjustment and consensus

The key insight: combining multiple signals outperforms any individual approach.

### Converting Games to Plate Appearances

Once you have a games-played estimate, [Zimmerman's formula](https://fantasy.fangraphs.com/estimating-playing-plate-appearances-knowing-team-talent/) converts to PA:

```
PA = (10.655 * team_RS_per_game + 705) - ((lineup_slot - 1) * 17.81)
```

Each lineup slot costs ~18 PA over a full season. The team offense effect is surprisingly small — only ~13 PA separates the best and worst offenses across a lineup.

## Emerging Research

### ML-Based Injury Prediction

A [2020 study (Karnuta et al.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7672741/) developed an ensemble of ML models (random forest, XGBoost, KNN) that achieved AUC = 0.76 for predicting next-season position player injuries. Key predictive features:

- Prior injury history (most important)
- Player age
- WAR / workload metrics
- Position-specific pitch data

ML outperformed logistic regression in 13 of 14 cases. However, even the best models only achieved "fair" reliability (70% accuracy for position players, 64% for pitchers), confirming the inherent difficulty.

### Joint Aging + Survival Models

A [FanGraphs community study](https://community.fangraphs.com/joint-model-of-the-war-aging-curve/) combined longitudinal aging models with Cox Proportional Hazards survival models. This corrects for **survivor bias** — players remaining in their 30s are disproportionately talented, skewing naive aging curves upward. Key findings:

- Batters peak at 26.5 (vs. 27.2 in naive models)
- Pitchers peak at 26.8 (vs. 27.7 in naive models)
- Joint modeling produces more realistic projections for young prospects

### Position-Specific Attrition

[BP research](https://www.baseballprospectus.com/news/article/23041/attrition-by-position-how-long-do-players-at-each-position-last/) using Cox Proportional Hazards models found significant position effects on career longevity:

- **Shortest careers**: DHs (~40% higher attrition risk) and 1B
- **Longest careers**: SS and C — they can move down the defensive spectrum when skills decline
- **Pitchers**: ~40% chance of IL stint in any given year; any trip can end a career

## The Layered Decomposition

The consensus best practice decomposes playing time into three layers:

1. **Roster spot** — Will the player have an MLB roster spot? (Human judgment, depth chart modeling, or transaction-aware rules)
2. **Games played** — How many games will they appear in? (Regression or ML model using age, injury history, prior PA, talent level, position)
3. **PA per game** — How many PA given their games? (Lineup position + team offense context)

No purely algorithmic system has matched the accuracy of the hybrid human-edited depth chart approach (RosterResource), which suggests the problem has a large irreducible component tied to managerial decisions, spring training battles, and transaction timing.

## Implications for This Project

Our playing time model (R² ≈ 0.22 batters, 0.26 pitchers) underperforms Marcel for top-300 players. This is consistent with the literature: point-estimate accuracy for established players is bounded by stochastic events. The Hardball Times model's R² = 0.74 relied heavily on a "starter status" variable (Opening Day starter), which is essentially human-provided depth chart information — the same thing RosterResource provides.

Rather than continuing to chase point-estimate accuracy with more features or model complexity, the highest-value path is:

1. **Import third-party depth chart projections** (FanGraphs Depth Charts) as the playing time foundation
2. **Use our distributional model** to add calibrated uncertainty intervals around those point estimates
3. **Reserve our first-party model** for players not covered by depth charts (minor leaguers, fringe roster players)

## Further Reading

- [A Guide to the Projection Systems](https://www.beyondtheboxscore.com/2016/2/22/11079186/projections-marcel-pecota-zips-steamer-explained-guide-math-is-fun) — Overview of Marcel, PECOTA, ZiPS, and Steamer
- [Projection Systems (FanGraphs Sabermetrics Library)](https://library.fangraphs.com/principles/projections/) — Canonical reference
- [How to Project Plate Appearances](https://www.smartfantasybaseball.com/2015/11/how-to-project-plate-appearances/) — Decomposing PA into games and lineup slot
- [Projecting Playing Time (Hardball Times)](https://tht.fangraphs.com/projecting-playing-time/) — The regression model with R² = 0.74
- [2024 Projection Review: Batter Playing Time](https://fantasy.fangraphs.com/2024-projection-review-batter-playing-time/) — Empirical accuracy comparison
- [Estimating Playing PA Knowing Team Talent](https://fantasy.fangraphs.com/estimating-playing-plate-appearances-knowing-team-talent/) — Zimmerman's lineup-slot formula
- [ML Outperforms Regression for Injury Prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC7672741/) — Karnuta et al. 2020
- [Joint Model of the WAR Aging Curve](https://community.fangraphs.com/joint-model-of-the-war-aging-curve/) — Aging + survival joint model
- [Attrition by Position](https://www.baseballprospectus.com/news/article/23041/attrition-by-position-how-long-do-players-at-each-position-last/) — Position-specific career lengths
- [The Delta Method, Revisited](https://www.baseballprospectus.com/news/article/59972/the-delta-method-revisited/) — Rethinking aging curves
- [PECOTA 2026 Updates and Challenges](https://www.baseballprospectus.com/news/article/104636/pecota-2026-updates-and-ongoing-challenges/) — Current state of PECOTA
