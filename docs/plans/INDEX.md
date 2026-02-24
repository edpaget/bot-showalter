# Roadmap Index

Overview of all active roadmaps, their status, and cross-roadmap dependencies.

## Active Roadmaps

| Roadmap | Phases | Progress | Hard Dependencies |
|---------|--------|----------|-------------------|
| [ADP Integration](adp-integration.md) | 5 | all phases done | none |
| [Breakout / Bust Classifier](breakout-bust-classifier.md) | 4 | not started | ADP, projections, valuations (all done) |
| [Category Balance Tracker](category-balance-tracker.md) | 3 | not started | none |
| [Composite GBM Tuning](composite-gbm-tuning.md) | 7 | phase 2 done | none |
| [Discord Bot](discord-bot.md) | 1 | not started | LLM Agent (all phases done) |
| [Draft Board Export](draft-board-export.md) | 3 | all phases done | valuations (done) |
| [Draft Pick Trade Evaluator](draft-pick-trade-evaluator.md) | 4 | not started | ADP (done), draft board (done) |
| [Injury Risk Discount](injury-risk-discount.md) | 3 | not started | none |
| [Keeper Optimization Solver](keeper-optimization-solver.md) | 4 | not started | keeper-surplus-value |
| [LLM Agent](llm-agent.md) | 3 | phases 1-2 done | none |
| [Player Eligibility](player-eligibility.md) | 3 | all phases done | none |
| [Principles Enforcement](principles-enforcement.md) | 6 | phases 1-3 done | none |
| [Keeper Surplus Value](keeper-surplus-value.md) | 4 | not started | none |
| [Live Draft Tracker](live-draft-tracker.md) | 4 | phases 1-2 done | valuations (done) |
| [Mock Draft Simulator](mock-draft-simulator.md) | 4 | not started | draft board (done), ADP (done) |
| [NPB/KBO Ingest](npb-kbo-ingest.md) | 6 | not started | none |
| [Positional Scarcity](positional-scarcity.md) | 3 | not started | none |
| [Positional Upgrade Calculator](positional-upgrade-calculator.md) | 4 | not started | draft board (done) |
| [Preseason Spine](preseason-spine.md) | 3 | not started | none |
| [Projection Confidence Report](projection-confidence-report.md) | 3 | phases 1-2 done | none |
| [Roster Optimizer](roster-optimizer.md) | 3 | not started | valuations (done) |
| [Schedule Matchup Analyzer](schedule-matchup-analyzer.md) | 4 | not started | none |
| [Test Performance](test-performance.md) | 4 | not started | none |
| [Tier Generator](tier-generator.md) | 3 | all phases done | none |
| [Top-300 Tuning](top-300-tuning.md) | 6 | phases 1-5 done | none |
| [Web UI Foundation](web-ui-foundation.md) | 3 | not started | none |
| [Yahoo Fantasy Integration](yahoo-fantasy-integration.md) | 5 | phases 1-2 done | live-draft-tracker, keeper-surplus-value |

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
- **ADP integration** (done) enhances: tier generator CLI, live draft tracker
- **Player eligibility** enhances: positional scarcity (depends on correct position assignments)
- **Positional scarcity** enhances: roster optimizer, keeper optimization solver
- **Category balance tracker** enhances: live draft tracker, mock draft simulator, positional upgrade calculator
- **Web UI foundation** enables: future web-based views for LLM chat, live draft tracker UI, charts/visualizations
