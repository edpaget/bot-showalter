# Valuation Model Roadmap

## Goal

Add a valuation model that transforms stat projections into dollar values and draft rankings. This is the first model in the system that is **not** a projection model — it consumes projections as input rather than historical stats. It validates that the `Model` protocol and registry are genuinely model-type-agnostic and establishes the patterns for non-projection models going forward.

The concrete methodology is SGP (Standing Gain Points), the standard approach for rotisserie-style fantasy baseball valuation. Given a set of projections and league settings, it computes each player's marginal value above replacement level, converts that to a dollar amount, and ranks players accordingly.

---

## Problem Statement

The system currently produces stat projections (Marcel, third-party imports) and can evaluate their accuracy, but has no way to answer the questions that matter for draft day:

- **How much is this player worth?** A player projected for 35 HR is worth more in a league where HR are scarce than in one where they're plentiful.
- **Who should I draft next?** Rankings depend on league format, roster construction, and position scarcity — not just raw projected stats.
- **How do different projection systems compare in value terms?** Marcel and Steamer might disagree on a player's HR total, but how much does that disagreement matter in dollars?

These questions require a second layer of modeling that sits on top of projections. The valuation model is that layer.

---

## Core Insight: Valuations Are a Different Kind of Model

Projection models transform **historical stats → future stats**. Valuation models transform **future stats → dollar values**. The key differences:

| | Projection model (Marcel) | Valuation model (SGP) |
|---|---|---|
| Input | Historical batting/pitching stats | Stat projections |
| Output | Projected stats per player | Dollar value + rank per player |
| Depends on league settings | No | Yes |
| Trainable | Potentially (ML models) | Yes — learns SGP denominators from historical standings |
| Evaluable | Compare projected stats to actuals | Compare predicted values to actual production value |
| Persists output as | `projection` rows | `valuation` rows (new) |

The `Model` protocol already accommodates this — it requires only `name`, `description`, `supported_operations`, and `artifact_type`. The capability protocols (`Predictable`, `Trainable`, etc.) and `PredictResult.predictions: list[dict[str, Any]]` are generic enough. What's missing is:

1. A way for a model to **consume projections as input** (constructor-injected `ProjectionRepo`)
2. A **domain type and persistence layer** for valuations
3. **League settings** configuration
4. **Valuation-specific evaluation** (comparing predicted values to actual production value)

---

## Design

### League Settings

League configuration determines replacement level, category weights, and positional scarcity. It belongs in the TOML config:

```toml
[league]
teams = 12
budget = 260
batting_categories = ["hr", "r", "rbi", "sb", "avg"]
pitching_categories = ["w", "sv", "so", "era", "whip"]
roster_batters = 14
roster_pitchers = 9
roster_util = 1          # utility slots (any batter)

[league.positions]
# minimum starters per position
c = 1
first_base = 1
second_base = 1
third_base = 1
ss = 1
of = 5
```

These flow into `ModelConfig.model_params` via the existing TOML-to-config pipeline, so the valuation model reads them from `config.model_params["league"]`.

### Valuation Domain Type

```python
@dataclass(frozen=True)
class Valuation:
    player_id: int
    season: int
    system: str               # valuation system, e.g. "sgp"
    version: str
    projection_system: str    # source projections, e.g. "marcel"
    projection_version: str
    player_type: str          # "batter" or "pitcher"
    position: str             # primary eligible position
    value: float              # dollar value
    rank: int                 # overall rank
    sgp_breakdown: dict[str, float]  # per-category SGP contributions
    id: int | None = None
```

### Valuation Table

```sql
CREATE TABLE valuation (
    id                   INTEGER PRIMARY KEY,
    player_id            INTEGER NOT NULL REFERENCES player(id),
    season               INTEGER NOT NULL,
    system               TEXT NOT NULL,
    version              TEXT NOT NULL,
    projection_system    TEXT NOT NULL,
    projection_version   TEXT NOT NULL,
    player_type          TEXT NOT NULL,
    position             TEXT NOT NULL,
    value                REAL NOT NULL,
    rank                 INTEGER NOT NULL,
    sgp_json             TEXT,
    UNIQUE(player_id, season, system, version)
);
```

### ValuationRepo

```python
@runtime_checkable
class ValuationRepo(Protocol):
    def upsert(self, valuation: Valuation) -> int: ...
    def get_by_player_season(self, player_id: int, season: int, system: str | None = None) -> list[Valuation]: ...
    def get_by_system_version(self, system: str, version: str) -> list[Valuation]: ...
    def get_rankings(self, system: str, version: str, player_type: str | None = None) -> list[Valuation]: ...
```

### SGP Engine (Pure Functions)

The SGP methodology is a pipeline of pure functions:

1. **Compute SGP denominators** — How many standings points does one marginal unit of each stat buy? Learned from historical standings data, or estimated from the projection pool itself.
2. **Compute replacement level** — The stat line of the best player who would go undrafted, given league size and roster constraints.
3. **Convert stats to SGP** — `(player_stat - replacement_stat) / sgp_denominator` per category.
4. **Compute dollar values** — Distribute the league's total budget proportionally across total SGP.
5. **Apply position scarcity** — Adjust values based on positional supply/demand.
6. **Rank** — Sort by adjusted dollar value.

```python
def compute_sgp_denominators(
    projections: list[dict[str, Any]],
    categories: list[str],
    num_teams: int,
) -> dict[str, float]: ...

def compute_replacement_level(
    projections: list[dict[str, Any]],
    categories: list[str],
    roster_spots: int,
) -> dict[str, float]: ...

def stats_to_sgp(
    stats: dict[str, float],
    replacement: dict[str, float],
    denominators: dict[str, float],
    negative_categories: set[str],
) -> dict[str, float]: ...

def sgp_to_dollars(
    player_sgp: list[tuple[int, float]],
    total_budget: float,
) -> list[tuple[int, float]]: ...
```

These are stateless functions that operate on plain dicts. The `SgpValuationModel` orchestrates calling them in sequence.

### SgpValuationModel

```python
@register("sgp")
class SgpValuationModel:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        player_repo: PlayerRepo,
    ) -> None: ...

    # Model protocol
    name = "sgp"
    description = "SGP-based auction valuation from stat projections"
    supported_operations = frozenset({"predict"})
    artifact_type = "none"

    def predict(self, config: ModelConfig) -> PredictResult:
        # 1. Read league settings from config.model_params
        # 2. Read projections for the specified system/version/season
        # 3. Run SGP pipeline
        # 4. Return PredictResult with value/rank dicts
        ...
```

The model takes `ProjectionRepo` and `PlayerRepo` as constructor dependencies — consistent with how Marcel takes `DatasetAssembler`. The composition root wires them in.

### Composition Root

A new branch in `create_model` and `build_model_context` for models that need repo access:

```python
model = create_model(model_name, assembler=assembler, projection_repo=projection_repo, player_repo=player_repo)
```

The existing `create_model` already filters kwargs to match the model's constructor signature, so passing extra kwargs that SGP doesn't need (like `assembler`) is harmless, and passing kwargs that Marcel doesn't need (like `projection_repo`) is equally harmless.

### Valuation-Specific Evaluation

`ProjectionEvaluator` can't be reused — it compares stat predictions to stat actuals. Valuation evaluation asks: **did the predicted dollar values match actual production value?**

The approach: compute "actual values" by running the same SGP methodology on end-of-season actual stats, then compare predicted values to actual values.

```python
class ValuationEvaluator:
    def __init__(
        self,
        valuation_repo: ValuationRepo,
        batting_repo: BattingStatsRepo,
        pitching_repo: PitchingStatsRepo,
    ) -> None: ...

    def evaluate(
        self,
        system: str,
        version: str,
        season: int,
    ) -> ValuationEvalResult: ...
```

Metrics:
- **Value MAE** — average absolute error in dollar values
- **Rank correlation** — Spearman correlation between predicted and actual rankings
- **Profit** — how much surplus value a drafter would have captured using these rankings vs. ADP

---

## Dependencies

- The `Model` protocol and registry (complete).
- At least one working projection system to consume (Marcel is complete).
- Player position data in the `player` table (need to verify this exists or add it).

---

## Phases

### Phase 1 — League settings and valuation domain types

- Add `Valuation` frozen dataclass to `domain/valuation.py`
- Add `valuation` table schema (migration)
- Add `ValuationRepo` protocol to `repos/protocols.py`
- Add `SqliteValuationRepo` implementation
- Extend TOML config to support a `[league]` section, surfaced as `model_params["league"]`
- Tests: repo CRUD with in-memory SQLite, config parsing

### Phase 2 — SGP engine

- Create `models/sgp/engine.py` with pure functions:
  - `compute_sgp_denominators` — estimate from the projection pool
  - `compute_replacement_level` — determine replacement-level stat line
  - `stats_to_sgp` — convert one player's stats to SGP
  - `sgp_to_dollars` — distribute budget across SGP
  - `apply_position_scarcity` — adjust for positional supply/demand
- Handle negative-is-good categories (ERA, WHIP) — these need inverted SGP
- Tests: known-input/known-output unit tests for each function, edge cases (empty pool, single player, ties)

### Phase 3 — SGP model registration and predict

- Create `models/sgp/model.py` with `SgpValuationModel`
  - Implements `Model`, `Predictable`
  - Constructor takes `ProjectionRepo`, `PlayerRepo`
  - `predict()` orchestrates: read projections → run SGP pipeline → build `PredictResult`
- Register as `@register("sgp")`
- Update composition root in `cli/factory.py` to pass repo dependencies
- Tests: end-to-end predict with `FakeProjectionRepo` and `FakePlayerRepo`

### Phase 4 — Valuation persistence and lookup

- After `predict()`, persist valuations to the `valuation` table
  - Either in the model's `predict()` or as a post-predict step in the dispatcher/CLI
- `ValuationLookupService` — look up a player's valuations, get rankings
- CLI commands:
  - `fbm valuations lookup "Soto" --season 2026` — player value across systems
  - `fbm valuations rankings --season 2026 --system sgp --version v1` — ranked list
- Output formatters in `_output.py`
- Tests: lookup service with fakes, CLI with `CliRunner`

### Phase 5 — Valuation evaluation

- `ValuationEvaluator` service
  - Compute "actual production values" from end-of-season stats using the same SGP methodology
  - Compare predicted values to actual values
  - Compute metrics: value MAE, rank correlation (Spearman), surplus value
- `ValuationEvalResult` domain type
- CLI: `fbm eval-valuations sgp --version v1 --season 2025`
- Tests: evaluator with known predicted/actual values

---

## Phase Order

```
Phase 1 (domain + persistence)
  ↓
Phase 2 (SGP engine)
  ↓
Phase 3 (model + predict)
  ↓
Phase 4 (persistence + lookup CLI)      Phase 5 (evaluation)
  [independent, both depend on Phase 3]
```

---

## Out of Scope

- **Points-league valuation** — SGP is for rotisserie leagues. Points-league valuation is simpler (linear weights) and can be added as a separate model later.
- **Draft simulation / auction optimizer** — Knowing player values is the input to draft strategy, but actually simulating a draft or recommending optimal bids is a separate problem.
- **ADP integration** — Comparing SGP values to Average Draft Position is useful but not required for the core valuation model.
- **Dynamic in-draft re-valuation** — Updating values as players are drafted off the board is an interactive feature, not a batch model.
