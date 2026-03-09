# Roadmap Index

Overview of all roadmaps, their status, and cross-roadmap dependencies.

## Active Roadmaps

| Roadmap | Phases | Progress | Hard Dependencies |
|---------|--------|----------|-------------------|
| [Keeper League Analysis](keeper-league-analysis.md) | 3 | phase 1 done | keeper-surplus-value (done), yahoo-fantasy-integration (done) |
| [Composite GBM Tuning](composite-gbm-tuning.md) | 7 | phase 2 done | none |
| [K8s Deployment](k8s-deployment.md) | 4 | not started | none |
| [NPB/KBO Ingest](npb-kbo-ingest.md) | 6 | not started | none |
| [Schedule Matchup Analyzer](schedule-matchup-analyzer.md) | 4 | not started | none |
| [Test Performance](test-performance.md) | 4 | not started | none |
| [ZAR Replacement-Padded](zar-replacement-padded.md) | 3 | phases 1-2 done | injury-risk-discount (done), valuation-system-unification phase 1 |
| [Web UI Foundation](web-ui-foundation.md) | 6 | phases 1-3 done | draft-session-persistence (phase 2 needs phases 1-2) |
| [Opponent Draft Model](opponent-draft-model.md) | 3 | not started | live-draft-tracker (done), adp-integration (done) |
| [ADP Arbitrage Alerts](adp-arbitrage-alerts.md) | 2 | not started | live-draft-tracker (done), adp-integration (done) |
| [Mock Draft Insights](mock-draft-insights.md) | 3 | not started | mock-draft-simulator (done), live-draft-tracker (done) |
| [CLI Consistency](cli-consistency.md) | 4 | phase 1 done | none |
| [Evaluation Framework](evaluation-framework.md) | 3 | not started | none |
| [Valuation Accuracy](valuation-accuracy.md) | 4 | phases 1-2 done | breakout-bust-classifier (done), injury-risk-discount (done), variance-correction (done) |

## Completed Roadmaps

| Roadmap | Phases |
|---------|--------|
| [Valuation Version Consistency](valuation-version-consistency.md) | 2 |
| [Actual Valuation SP/RP Split](actual-valuation-sp-rp.md) | 1 |
| [Draft Session Persistence](draft-session-persistence.md) | 3 |
| [Yahoo Draft Ingestion Fixes](yahoo-draft-ingestion-fixes.md) | 3 |
| [ADP Integration](adp-integration.md) | 5 |
| [Data Profiling Tools](data-profiling-tools.md) | 3 |
| [Category Balance Tracker](category-balance-tracker.md) | 3 |
| [Discord Bot](discord-bot.md) | 1 |
| [Evaluation Guardrails](evaluation-guardrails.md) | 4 |
| [Experiment Journal](experiment-journal.md) | 3 |
| [LLM Agent](llm-agent.md) | 3 |
| [Draft Board Export](draft-board-export.md) | 3 |
| [Draft Pick Keeper Support](draft-pick-keeper-support.md) | 3 |
| [Draft Pick Trade Evaluator](draft-pick-trade-evaluator.md) | 4 |
| [Live Draft Tracker](live-draft-tracker.md) | 4 |
| [Player Eligibility](player-eligibility.md) | 3 |
| [Projection Confidence Report](projection-confidence-report.md) | 3 |
| [Preseason Spine](preseason-spine.md) | 3 |
| [Principles Enforcement](principles-enforcement.md) | 6 |
| [Pitcher Calibration](pitcher-calibration.md) | 4 |
| [Keeper Optimization Solver](keeper-optimization-solver.md) | 4 |
| [Keeper Surplus Value](keeper-surplus-value.md) | 4 |
| [Mock Draft Simulator](mock-draft-simulator.md) | 4 |
| [Positional Scarcity](positional-scarcity.md) | 3 |
| [Positional Upgrade Calculator](positional-upgrade-calculator.md) | 4 |
| [Roster Optimizer](roster-optimizer.md) | 3 |
| [Tier Generator](tier-generator.md) | 3 |
| [Fast Feedback Loop](fast-feedback-loop.md) | 3 |
| [Feature Candidate Factory](feature-candidate-factory.md) | 3 |
| [Top-300 Tuning](top-300-tuning.md) | 6 |
| [Residual Analysis Tools](residual-analysis-tools.md) | 3 |
| [Validation Gate](validation-gate.md) | 2 |
| [Yahoo Architecture Fixes](yahoo-architecture-fixes.md) | 3 |
| [Yahoo Eligibility Rules](yahoo-eligibility-rules.md) | 2 |
| [Injury Model Fixes](injury-model-fixes.md) | 3 |
| [Injury Risk Discount](injury-risk-discount.md) | 3 |
| [Injury Valuation Cleanup](injury-valuation-cleanup.md) | 3 |
| [Keeper League Analysis](keeper-league-analysis.md) | 3 |
| [Test Coverage](test-coverage.md) | 5 |
| [Projection Blender](projection-blender.md) | 4 |
| [Yahoo Fantasy Integration](yahoo-fantasy-integration.md) | 5 |
| [Position Normalization](position-normalization.md) | 3 |
| [Yahoo Live Draft Fixes](yahoo-live-draft-fixes.md) | 3 |
| [Breakout / Bust Classifier](breakout-bust-classifier.md) | 4 |
| [Connection Pool Repos](connection-pool-repos.md) | 4 |
| [FanGraphs Projection Sync](fangraphs-projection-sync.md) | 3 |
| [Experiment System Generalization](experiment-system-generalization.md) | 4 |
| [Model Training Inspector](model-training-inspector.md) | 3 |
| [Player Bio Fuzzy Team](player-bio-fuzzy-team.md) | 3 |
| [Roster Stint Preload](roster-stint-preload.md) | 2 |
| [Playing Time Flexibility](playing-time-flexibility.md) | 4 |
| [Prior-Season Roster Lookup](prior-season-roster-lookup.md) | 3 |
| [Valuation System Unification](valuation-system-unification.md) | 2 |
| [Variance Correction](variance-correction.md) | 3 |
| [Web API Hardening](web-api-hardening.md) | 3 |
| [Yahoo Integration Improvements](yahoo-integration-improvements.md) | 3 |
| [ZAR Distributional](zar-distributional.md) | 3 |
| [ZAR Replacement-Padded](zar-replacement-padded.md) | 3 |

Older completed roadmaps are in the [`archive/`](archive/) directory.

## Dependency Graph

Roadmaps that must complete (at least partially) before others can start.

```
keeper-surplus-value ──► keeper-optimization-solver
                    ──► yahoo-fantasy-integration

live-draft-tracker ────► yahoo-fantasy-integration

data-profiling-tools (done) ──► feature-candidate-factory (--correlate flag)

fast-feedback-loop ──► validation-gate (CV infrastructure)
```

mock-draft-simulator ─────► mock-draft-insights (simulation data)

All other dependencies (valuations, ADP, draft board, projections, live-draft-tracker, mock-draft-simulator) are already satisfied by completed work.

## Optional Integrations

These aren't hard blockers but enhance the consuming roadmap when available:

- **Tier generator** enhances: draft board export, live draft tracker, mock draft simulator, positional upgrade calculator, roster optimizer
- **ADP integration** (done) enhances: tier generator CLI, live draft tracker
- **Player eligibility** enhances: positional scarcity (depends on correct position assignments)
- **Positional scarcity** enhances: roster optimizer, keeper optimization solver
- **Category balance tracker** (done) enhances: live draft tracker, mock draft simulator, positional upgrade calculator
- **Web UI foundation** enables: future web-based views for LLM chat, live draft tracker UI, charts/visualizations
- **Data profiling tools** (done) enhances: feature-candidate-factory (--correlate flag chains into target correlation scanner)
- **Feature candidate factory** enhances: fast-feedback-loop (--inject flag uses named candidates)
- **Fast feedback loop** enhances: experiment-journal (auto-logging from quick-eval/marginal-value), validation-gate (shared CV infrastructure)
- **Experiment journal** enhances: validation-gate (auto-log full validation results)
- **Playing time flexibility** enhances: projection-blender (flexible PT source for blended projections)
- **Opponent draft model** enhances: live-draft-tracker (threat/run alerts in REPL), mock-draft-insights (opponent behavior modeling)
- **ADP arbitrage alerts** enhances: live-draft-tracker (falling player alerts in REPL)
- **Mock draft insights** enhances: live-draft-tracker (mock-informed recommendations)
