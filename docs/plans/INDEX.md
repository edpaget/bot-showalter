# Roadmap Index

Overview of all roadmaps, their status, and cross-roadmap dependencies.

## Active Roadmaps

| Roadmap | Phases | Progress | Hard Dependencies |
|---------|--------|----------|-------------------|
| [Breakout / Bust Classifier](breakout-bust-classifier.md) | 4 | not started | ADP, projections, valuations (all done) |
| [Composite GBM Tuning](composite-gbm-tuning.md) | 7 | phase 2 done | none |
| [Draft Pick Keeper Support](draft-pick-keeper-support.md) | 3 | not started | draft-pick-trade-evaluator phase 1, keeper-surplus-value (done) |
| [Draft Pick Trade Evaluator](draft-pick-trade-evaluator.md) | 4 | phase 1 done | ADP (done), draft board (done) |
| [Injury Risk Discount](injury-risk-discount.md) | 3 | not started | none |
| [Keeper Optimization Solver](keeper-optimization-solver.md) | 4 | not started | keeper-surplus-value |
| [Mock Draft Simulator](mock-draft-simulator.md) | 4 | not started | draft board (done), ADP (done) |
| [NPB/KBO Ingest](npb-kbo-ingest.md) | 6 | not started | none |
| [Positional Scarcity](positional-scarcity.md) | 3 | not started | none |
| [Positional Upgrade Calculator](positional-upgrade-calculator.md) | 4 | not started | draft board (done) |
| [Roster Optimizer](roster-optimizer.md) | 3 | not started | valuations (done) |
| [Schedule Matchup Analyzer](schedule-matchup-analyzer.md) | 4 | not started | none |
| [Test Performance](test-performance.md) | 4 | not started | none |
| [Web UI Foundation](web-ui-foundation.md) | 3 | not started | none |
| [Yahoo Fantasy Integration](yahoo-fantasy-integration.md) | 5 | phases 1-3 done | live-draft-tracker, keeper-surplus-value |

## Completed Roadmaps

| Roadmap | Phases |
|---------|--------|
| [ADP Integration](adp-integration.md) | 5 |
| [Category Balance Tracker](category-balance-tracker.md) | 3 |
| [Discord Bot](discord-bot.md) | 1 |
| [Evaluation Guardrails](evaluation-guardrails.md) | 4 |
| [LLM Agent](llm-agent.md) | 3 |
| [Draft Board Export](draft-board-export.md) | 3 |
| [Live Draft Tracker](live-draft-tracker.md) | 4 |
| [Player Eligibility](player-eligibility.md) | 3 |
| [Projection Confidence Report](projection-confidence-report.md) | 3 |
| [Preseason Spine](preseason-spine.md) | 3 |
| [Principles Enforcement](principles-enforcement.md) | 6 |
| [Pitcher Calibration](pitcher-calibration.md) | 4 |
| [Keeper Surplus Value](keeper-surplus-value.md) | 4 |
| [Tier Generator](tier-generator.md) | 3 |
| [Top-300 Tuning](top-300-tuning.md) | 6 |

Older completed roadmaps are in the [`archive/`](archive/) directory.

## Dependency Graph

Roadmaps that must complete (at least partially) before others can start.

```
keeper-surplus-value ──► keeper-optimization-solver
                    ──► yahoo-fantasy-integration

draft-pick-trade-evaluator (phase 1) ──► draft-pick-keeper-support
keeper-optimization-solver (phase 1) ──► draft-pick-keeper-support (phase 3)

live-draft-tracker ────► yahoo-fantasy-integration
```

All other dependencies (valuations, ADP, draft board, projections) are already satisfied by completed work.

## Optional Integrations

These aren't hard blockers but enhance the consuming roadmap when available:

- **Tier generator** enhances: draft board export, live draft tracker, mock draft simulator, positional upgrade calculator, roster optimizer
- **ADP integration** (done) enhances: tier generator CLI, live draft tracker
- **Player eligibility** enhances: positional scarcity (depends on correct position assignments)
- **Positional scarcity** enhances: roster optimizer, keeper optimization solver
- **Category balance tracker** (done) enhances: live draft tracker, mock draft simulator, positional upgrade calculator
- **Web UI foundation** enables: future web-based views for LLM chat, live draft tracker UI, charts/visualizations
