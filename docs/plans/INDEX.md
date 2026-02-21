# Roadmap Index

Overview of all active roadmaps, their status, and cross-roadmap dependencies.

## Active Roadmaps

| Roadmap | Phases | Progress | Hard Dependencies |
|---------|--------|----------|-------------------|
| [ADP Integration](adp-integration.md) | 5 | phases 1-2 done | none |
| [Breakout / Bust Classifier](breakout-bust-classifier.md) | 4 | not started | ADP, projections, valuations (all done) |
| [Category Balance Tracker](category-balance-tracker.md) | 3 | not started | none |
| [Composite GBM Tuning](composite-gbm-tuning.md) | 7 | phase 2 done | none |
| [Draft Board Export](draft-board-export.md) | 3 | all phases done | valuations (done) |
| [Draft Pick Trade Evaluator](draft-pick-trade-evaluator.md) | 4 | not started | ADP (done), draft board (done) |
| [Injury Risk Discount](injury-risk-discount.md) | 3 | not started | none |
| [Keeper Optimization Solver](keeper-optimization-solver.md) | 4 | not started | keeper-surplus-value |
| [LLM Agent](llm-agent.md) | 3 | phase 1 done | none |
| [Keeper Surplus Value](keeper-surplus-value.md) | 4 | not started | none |
| [Live Draft Tracker](live-draft-tracker.md) | 4 | not started | valuations (done) |
| [Mock Draft Simulator](mock-draft-simulator.md) | 4 | not started | draft board (done), ADP (done) |
| [NPB/KBO Ingest](npb-kbo-ingest.md) | 6 | not started | none |
| [Positional Scarcity](positional-scarcity.md) | 3 | not started | none |
| [Positional Upgrade Calculator](positional-upgrade-calculator.md) | 4 | not started | draft board (done) |
| [Projection Confidence Report](projection-confidence-report.md) | 3 | phases 1-2 done | none |
| [Roster Optimizer](roster-optimizer.md) | 3 | not started | valuations (done) |
| [Schedule Matchup Analyzer](schedule-matchup-analyzer.md) | 4 | not started | none |
| [Tier Generator](tier-generator.md) | 3 | phase 1 done | none |
| [Top-300 Tuning](top-300-tuning.md) | 6 | phase 1 done | none |
| [Yahoo Fantasy Integration](yahoo-fantasy-integration.md) | 5 | not started | live-draft-tracker, keeper-surplus-value |

## Dependency Graph

Roadmaps that must complete (at least partially) before others can start.

```
keeper-surplus-value ──► keeper-optimization-solver
                    ──► yahoo-fantasy-integration

live-draft-tracker ────► yahoo-fantasy-integration
```

All other dependencies (valuations, ADP, draft board, projections) are already satisfied by completed work.

## Optional Integrations

These aren't hard blockers but enhance the consuming roadmap when available:

- **Tier generator** enhances: draft board export, live draft tracker, mock draft simulator, positional upgrade calculator, roster optimizer
- **ADP integration** (remaining phases) enhances: tier generator CLI, live draft tracker
- **Positional scarcity** enhances: roster optimizer, keeper optimization solver
- **Category balance tracker** enhances: live draft tracker, mock draft simulator, positional upgrade calculator
