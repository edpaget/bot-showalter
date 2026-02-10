# Contextual Model Bug Audit

Systematic code review of the contextual transformer implementation. Captures bugs, conceptual errors, and fragile design patterns found via static analysis.

Related: [contextual-model-evaluation.md](contextual-model-evaluation.md) (performance analysis and improvement plan)

---

## Critical Issues

### C1. Numeric feature normalization is statistically wrong

- [x] **Fixed**

**File:** `src/fantasy_baseball_manager/contextual/model/embedder.py:93-100`

The `EventEmbedder` computes per-position instance normalization **across the 23 numeric features**:

```python
mask_float = numeric_mask.float()
n_present = mask_float.sum(dim=-1, keepdim=True).clamp(min=1)
masked_numeric = numeric_features * mask_float
mean = masked_numeric.sum(dim=-1, keepdim=True) / n_present
var = ((numeric_features - mean).pow(2) * mask_float).sum(dim=-1, keepdim=True) / n_present
normed = (numeric_features - mean) / (var + self.numeric_norm.eps).sqrt()
normed = normed * self.numeric_norm.weight + self.numeric_norm.bias
normed = normed * mask_float
```

This computes a single mean and variance over all 23 features at each token position. These features have vastly different scales and semantics:

| Feature | Typical range |
|---------|--------------|
| release_speed | 70-100 |
| release_spin_rate | 1500-3000 |
| balls | 0-3 |
| strikes | 0-2 |
| delta_run_exp | -2 to +2 |
| is_lhb (boolean) | 0-1 |

Computing mean/variance across these is meaningless — it normalizes `release_speed=95` and `balls=3` together in a single statistic. The correct approach is **per-feature normalization using population statistics** computed from the training set (i.e., standard feature-wise standardization).

The `nn.LayerNorm` affine parameters partially compensate, but the underlying normalization destroys per-feature scale information before the affine transform can recover it.

**Fix:** Replace with per-feature normalization using precomputed training-set mean/std per feature, or use a proper `LayerNorm` over the feature dimension with correct semantics.

---

### C2. Hardcoded default `ModelConfig()` in predictor

- [x] **Fixed**

**File:** `src/fantasy_baseball_manager/contextual/predictor.py:51, 168`

```python
model_config = ModelConfig()
```

The predictor always instantiates `ModelConfig()` with defaults (d_model=256, n_layers=4, n_heads=8, ff_dim=1024). If the model was trained with non-default architecture flags (e.g., `--d-model 64 --n-layers 2`), `load_state_dict` will fail at inference time with a tensor shape mismatch.

The CLI fine-tune command has the same issue — it auto-detects `max_seq_len` from the checkpoint but expects the user to manually pass matching architecture flags.

**Fix:** Persist `ModelConfig` as part of `ContextualModelMetadata` at save time. Load it back and use it to reconstruct the model at inference time.

---

### C3. Pre-training task is too easy due to visible `pa_event`

- [x] **Fixed**

**File:** `src/fantasy_baseball_manager/contextual/training/dataset.py` (MGM masking logic)

During masked gamestate modeling, only `pitch_type` and `pitch_result` are masked. The `pa_event` categorical feature remains visible. Since `pa_event` encodes the plate appearance outcome (e.g., "strikeout", "home_run", "walk"), the model can trivially infer masked targets:

- `pa_event="strikeout"` + last pitch in PA -> `pitch_result` is almost certainly `swinging_strike` or `called_strike`
- `pa_event="home_run"` -> `pitch_result="hit_into_play"`, `pitch_type` correlates with hittable pitches

This shortcut undermines pre-training. The model learns to read `pa_event` rather than building rich contextual representations from pitch sequences. The result is weaker transfer to fine-tuning.

**Fix:** Mask `pa_event` on the same positions where `pitch_type` and `pitch_result` are masked, or mask it independently as an additional pre-training objective.

---

## Moderate Issues

### M1. `ignore_index=0` in pre-training loss couples PAD semantics with loss sentinel

- [x] **Fixed**

**File:** `src/fantasy_baseball_manager/contextual/training/pretrain.py:349-358`

```python
pt_loss = F.cross_entropy(
    pt_logits.view(-1, pt_logits.size(-1)),
    target_pt.view(-1),
    ignore_index=0,
)
```

Non-masked positions use 0 as a "skip this" sentinel in the target tensor. PAD is also index 0 in all vocabularies. This works by coincidence but is fragile — if any vocabulary ever assigned a real token to index 0, it would be silently ignored during training.

**Fix:** Use a dedicated sentinel value (e.g., -100, PyTorch's default `ignore_index`) for non-masked positions.

---

### M2. Player token self-attention works by accident

- [x] **Fixed**

**Files:** `src/fantasy_baseball_manager/contextual/model/mask.py:52-57`, `src/fantasy_baseball_manager/contextual/model/transformer.py:76`

Player tokens are designed to attend only to same-game pitch tokens — they explicitly do **not** attend to themselves in `build_player_attention_mask`:

```python
player_can_attend = player_rows & same_game & non_player_real_cols
```

Self-attention only works because of a NaN-prevention diagonal override in the transformer:

```python
float_mask[:, diag, diag] = 0.0
```

This patch was added for padding positions but has the side effect of enabling self-attention for all positions, including player tokens. The behavior is correct but the intent is wrong — a future refactor that removes the diagonal hack would silently break player token self-attention.

**Fix:** Explicitly include the diagonal in `player_can_attend` in `mask.py`.

---

### M3. CLS token initial embedding is identical to PAD

- [x] **Fixed**

**File:** `src/fantasy_baseball_manager/contextual/model/tensorizer.py:124-138`

The [CLS] token has all-zero categorical IDs and zero numeric features with all-False mask — the exact same representation as padding tokens. The model must rely entirely on positional encoding (position 0) to differentiate CLS from PAD.

This is a weak signal. Standard BERT implementations use a dedicated learned [CLS] embedding.

**Fix:** Add a learned CLS embedding vector, or use a distinct token type embedding to differentiate CLS from PAD.

---

### M4. Fine-tune MSE loss has no per-stat weighting

- [x] **Fixed**

**File:** `src/fantasy_baseball_manager/contextual/training/finetune.py:239`

```python
loss = F.mse_loss(preds, batch.targets)
```

Raw MSE across all target stats treats them equally. HR rates (~0.03) have much smaller variance than SO rates (~0.22), so the gradient signal is dominated by high-variance stats. The model under-optimizes for rare events.

For count-mode targets the imbalance is even worse: SO counts (~6/game for pitchers) dominate HR counts (~1/game).

**Fix:** Normalize targets by per-stat training-set std before computing MSE, or use per-stat loss weights inversely proportional to variance.

---

## Minor Issues

### m1. No gradient accumulation support

- [x] **Fixed**

**Files:** `src/fantasy_baseball_manager/contextual/training/pretrain.py`, `finetune.py`

Both trainers call `optimizer.zero_grad()` and `optimizer.step()` on every batch. For long sequences requiring small batch sizes, effective batch size cannot be increased without gradient accumulation.

---

### m2. `extract_game_stats` PA boundary heuristic is data-format dependent

- [x] **Fixed**

**File:** `src/fantasy_baseball_manager/contextual/training/dataset.py:379-392`

```python
is_last_in_pa = (
    idx == n_pitches - 1
    or pitches[idx + 1].pitch_number == 1
)
```

Assumes `pitch_number` resets to 1 at the start of each plate appearance. Some Statcast data sources use cumulative pitch numbering within a game. If the data format changes, this silently double-counts or misattributes events.

---

### m3. `game_id=0` for padding overlaps with first game's ID

- [x] **Fixed**

**File:** `src/fantasy_baseball_manager/contextual/model/tensorizer.py` (collate function)

`game_ids` is initialized with `torch.zeros(...)`. Padding positions get `game_id=0`, the same value as the first game's pitch tokens. This is masked out in the attention logic, but the semantic overlap is confusing and could cause bugs if mask logic is modified.

**Fix:** Initialize padding game_ids to -2 or another unused sentinel.
