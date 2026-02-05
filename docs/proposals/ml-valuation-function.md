# ML-Based Valuation Function

## What

Replace the fixed equal-weight z-score valuation with a neural network that learns how to convert stat projections into fantasy value. The model learns the market's valuation function from consensus projections and ADP, then applies that learned function to our projections to surface genuine projection disagreements.

## Why

The current z-score valuation treats all categories equally, but analysis of our top-50 rankings vs Yahoo ADP reveals systematic biases:

- **Speed players overvalued:** Perdomo (+43), Turang (+29), Abrams (+52), Hoerner (+55), Garcia (+40) — all high-SB, high-OBP, low-power profiles rank much higher in our system than ADP
- **Young upside players undervalued:** Caminero (-23), Chourio (-16), Kurtz (-13) — prospects with growth projections rank lower than ADP
- **Category correlations ignored:** HR/R/RBI are correlated (power hitters score in all three), while SB often anti-correlates with power. Equal weighting doesn't account for this.

The core problem is that our valuation function and the market's valuation function differ, which conflates two sources of disagreement:

1. **Projection disagreements** — we think a player will produce different stats than consensus (valuable alpha)
2. **Valuation disagreements** — we weight categories differently than the market (less actionable)

By learning the market's valuation function, we can isolate pure projection disagreements.

## Approach

### Separation of Concerns

The key insight is to train on consensus projections (Steamer/ZiPS) paired with ADP, then apply the learned function to our Marcel projections:

**Training:** Consensus projections → Neural Net → ADP (learns market valuation)

**Inference:** Our projections → Trained Net → Our value rankings

When we compare our output rankings to ADP, the differences now reflect only projection disagreements, since the valuation function is shared.

### Model Architecture

The network takes as input:

- Counting stats (HR, R, RBI, SB, W, K, SV)
- Rate stats (OBP, ERA, WHIP)
- Playing time indicators (PA, IP)
- Position encoding
- Strategy embedding

Shared dense layers learn stat interactions (e.g., diminishing returns on correlated stats, SB value conditional on OBP). A strategy-aware layer modulates the learned representations based on the selected draft strategy.

Output is a single value representing fantasy worth, trained to predict ADP rank or auction dollar value.

### Strategy Conditioning

Rather than training separate models per strategy, a single model accepts a strategy embedding as input. This allows:

- **Balanced:** Standard valuation learned from overall ADP
- **Power-focused:** Learned from ADP in formats that reward power
- **Speed-focused:** Emphasizes SB contribution in the loss function
- **Punt-saves:** Reduces closer value to near-zero

Strategy embeddings can be learned end-to-end if we have ADP data from different league formats (NFBC, Yahoo, ESPN, points leagues), or manually specified to shift category emphasis.

## Expected Impact

- Rankings that surface genuine projection alpha rather than valuation quirks
- Configurable strategies that adjust valuation without manual weight tuning
- Better understanding of which stat interactions matter (via learned weights)
- Reduced systematic bias toward speed-first players

## Pipeline Fit

New module: `valuation/ml_valuation.py`

The ML valuator implements the same interface as the existing z-score valuator, returning `PlayerValue` objects. It slots into the draft ranking pipeline as an alternative valuation method selectable via configuration.

Training is a separate offline step that produces model weights. Inference at draft time loads the trained model and runs forward passes on our projection set.

## Data Requirements

| Data | Source | Purpose |
|------|--------|---------|
| Steamer projections | FanGraphs | Training inputs |
| ZiPS projections | FanGraphs | Training inputs (alternative) |
| Yahoo ADP | Yahoo Fantasy API | Training targets |
| ESPN ADP | ESPN Fantasy API | Training targets (alternative) |
| NFBC ADP | NFBC | Training targets (deep leagues) |
| Our Marcel projections | Internal | Inference inputs |
| League format metadata | Manual | Strategy labels |

Historical data from 2-3 seasons provides sufficient training examples (~1500 players × 3 years = 4500 samples).

## Key Considerations

### Avoiding ADP Reproduction

If we trained on our own projections targeting ADP, the model would learn to reproduce ADP by implicitly learning the delta between our projections and consensus. By training on consensus projections, we force the model to learn only the valuation function. Our projection disagreements then surface naturally at inference time.

### Interpretability

While the full network is less interpretable than linear weights, we can extract insights via:

- Gradient-based feature importance
- SHAP values for individual predictions
- Probing the strategy embeddings to see which categories they emphasize

### Generalization

The model should generalize across seasons if stat distributions remain stable. Major structural changes (like the 2023 SB rule changes) may require retraining or adding era-aware features.

## Implementation Sketch

1. Build a data pipeline to fetch and align Steamer projections with ADP
2. Define the model architecture in PyTorch with configurable strategy embeddings
3. Train with MSE loss on ADP rank (or cross-entropy on binned value tiers)
4. Evaluate on held-out season to measure generalization
5. Integrate trained model into draft ranking pipeline as alternative valuator
6. Add strategy selection to CLI/UI for switching valuation modes
