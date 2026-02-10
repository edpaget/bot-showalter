"""Converts PitchEvent dataclasses into batched tensors for the model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.models import PitchEvent, PlayerContext
    from fantasy_baseball_manager.contextual.data.vocab import Vocabulary
    from fantasy_baseball_manager.contextual.model.config import ModelConfig

# Ordered list of numeric fields extracted from PitchEvent.
# Index position here determines column position in the numeric tensor.
CLS_GAME_ID: int = -1
PAD_GAME_ID: int = -2

NUMERIC_FIELDS: tuple[str, ...] = (
    "release_speed",  # 0  nullable
    "release_spin_rate",  # 1  nullable
    "pfx_x",  # 2  nullable
    "pfx_z",  # 3  nullable
    "plate_x",  # 4  nullable
    "plate_z",  # 5  nullable
    "release_extension",  # 6  nullable
    "launch_speed",  # 7  nullable
    "launch_angle",  # 8  nullable
    "hit_distance",  # 9  nullable
    "estimated_woba",  # 10 nullable
    "inning",  # 11
    "outs",  # 12
    "balls",  # 13
    "strikes",  # 14
    "bat_score",  # 15
    "fld_score",  # 16
    "pitch_number",  # 17
    "is_top",  # 18 bool→float
    "runners_on_1b",  # 19 bool→float
    "runners_on_2b",  # 20 bool→float
    "runners_on_3b",  # 21 bool→float
    "delta_run_exp",  # 22 nullable
)


@dataclass
class TensorizedSingle:
    """Tensorized representation of a single PlayerContext (no batch dim)."""

    pitch_type_ids: torch.Tensor  # (seq_len,) long
    pitch_result_ids: torch.Tensor  # (seq_len,) long
    bb_type_ids: torch.Tensor  # (seq_len,) long
    stand_ids: torch.Tensor  # (seq_len,) long
    p_throws_ids: torch.Tensor  # (seq_len,) long
    pa_event_ids: torch.Tensor  # (seq_len,) long
    numeric_features: torch.Tensor  # (seq_len, 23) float
    numeric_mask: torch.Tensor  # (seq_len, 23) bool
    padding_mask: torch.Tensor  # (seq_len,) bool — True=real
    player_token_mask: torch.Tensor  # (seq_len,) bool — True=player slot
    game_ids: torch.Tensor  # (seq_len,) long
    seq_length: int = field(default=0)


@dataclass
class TensorizedBatch:
    """Collated batch of TensorizedSingle instances with leading batch dim."""

    pitch_type_ids: torch.Tensor  # (batch, seq_len) long
    pitch_result_ids: torch.Tensor  # (batch, seq_len) long
    bb_type_ids: torch.Tensor  # (batch, seq_len) long
    stand_ids: torch.Tensor  # (batch, seq_len) long
    p_throws_ids: torch.Tensor  # (batch, seq_len) long
    pa_event_ids: torch.Tensor  # (batch, seq_len) long
    numeric_features: torch.Tensor  # (batch, seq_len, 23) float
    numeric_mask: torch.Tensor  # (batch, seq_len, 23) bool
    padding_mask: torch.Tensor  # (batch, seq_len) bool
    player_token_mask: torch.Tensor  # (batch, seq_len) bool
    game_ids: torch.Tensor  # (batch, seq_len) long
    seq_lengths: torch.Tensor = field(default_factory=lambda: torch.tensor([]))  # (batch,) long


class Tensorizer:
    """Converts PlayerContext dataclasses into tensor representations.

    Prepends a [CLS] aggregation token at position 0, then inserts a
    [PLAYER] token at the start of each game's pitch sequence.  CLS
    attends to all tokens across all games and is used for prediction
    during fine-tuning.  Categorical fields are encoded via the provided
    vocabularies.  Numeric fields are extracted in a fixed order (see
    NUMERIC_FIELDS).
    """

    def __init__(
        self,
        config: ModelConfig,
        pitch_type_vocab: Vocabulary,
        pitch_result_vocab: Vocabulary,
        bb_type_vocab: Vocabulary,
        handedness_vocab: Vocabulary,
        pa_event_vocab: Vocabulary,
    ) -> None:
        self._config = config
        self._pitch_type_vocab = pitch_type_vocab
        self._pitch_result_vocab = pitch_result_vocab
        self._bb_type_vocab = bb_type_vocab
        self._handedness_vocab = handedness_vocab
        self._pa_event_vocab = pa_event_vocab

    def tensorize_context(self, context: PlayerContext) -> TensorizedSingle:
        """Convert a PlayerContext into a TensorizedSingle."""
        pitch_type_ids: list[int] = []
        pitch_result_ids: list[int] = []
        bb_type_ids: list[int] = []
        stand_ids: list[int] = []
        p_throws_ids: list[int] = []
        pa_event_ids: list[int] = []
        numeric_features: list[list[float]] = []
        numeric_mask: list[list[bool]] = []
        player_token_mask: list[bool] = []
        game_id_list: list[int] = []

        n_numeric = len(NUMERIC_FIELDS)

        # Determine which games fit within max_seq_len (keep newest, drop oldest)
        # Account for the CLS token at position 0
        games = list(context.games)
        game_lengths = [1 + len(g.pitches) for g in games]  # 1 for player token

        total = 1 + sum(game_lengths)  # +1 for CLS
        while total > self._config.max_seq_len and games:
            total -= game_lengths[0]
            games.pop(0)
            game_lengths.pop(0)

        # Insert [CLS] token at position 0 — attends to all tokens across all games
        pitch_type_ids.append(0)
        pitch_result_ids.append(0)
        bb_type_ids.append(0)
        stand_ids.append(0)
        p_throws_ids.append(0)
        pa_event_ids.append(0)
        numeric_features.append([0.0] * n_numeric)
        numeric_mask.append([False] * n_numeric)
        player_token_mask.append(False)  # NOT a player token — gets pitch-like attention
        game_id_list.append(CLS_GAME_ID)

        for game_idx, game in enumerate(games):
            # Insert [PLAYER] token
            pitch_type_ids.append(0)
            pitch_result_ids.append(0)
            bb_type_ids.append(0)
            stand_ids.append(0)
            p_throws_ids.append(0)
            pa_event_ids.append(0)
            numeric_features.append([0.0] * n_numeric)
            numeric_mask.append([False] * n_numeric)
            player_token_mask.append(True)
            game_id_list.append(game_idx)

            # Encode each pitch
            for pitch in game.pitches:
                pitch_type_ids.append(self._pitch_type_vocab.encode(pitch.pitch_type))
                pitch_result_ids.append(self._pitch_result_vocab.encode(pitch.pitch_result))
                bb_type_ids.append(self._bb_type_vocab.encode(pitch.bb_type) if pitch.bb_type is not None else 0)
                stand_ids.append(self._handedness_vocab.encode(pitch.stand))
                p_throws_ids.append(self._handedness_vocab.encode(pitch.p_throws))
                pa_event_ids.append(self._pa_event_vocab.encode(pitch.pa_event) if pitch.pa_event is not None else 0)

                # Numeric features
                nums, mask = self._encode_numeric(pitch)
                numeric_features.append(nums)
                numeric_mask.append(mask)
                player_token_mask.append(False)
                game_id_list.append(game_idx)

        seq_len = len(pitch_type_ids)
        return TensorizedSingle(
            pitch_type_ids=torch.tensor(pitch_type_ids, dtype=torch.long),
            pitch_result_ids=torch.tensor(pitch_result_ids, dtype=torch.long),
            bb_type_ids=torch.tensor(bb_type_ids, dtype=torch.long),
            stand_ids=torch.tensor(stand_ids, dtype=torch.long),
            p_throws_ids=torch.tensor(p_throws_ids, dtype=torch.long),
            pa_event_ids=torch.tensor(pa_event_ids, dtype=torch.long),
            numeric_features=torch.tensor(numeric_features, dtype=torch.float32),
            numeric_mask=torch.tensor(numeric_mask, dtype=torch.bool),
            padding_mask=torch.ones(seq_len, dtype=torch.bool),
            player_token_mask=torch.tensor(player_token_mask, dtype=torch.bool),
            game_ids=torch.tensor(game_id_list, dtype=torch.long),
            seq_length=seq_len,
        )

    def collate(self, items: list[TensorizedSingle]) -> TensorizedBatch:
        """Collate a list of TensorizedSingle into a padded TensorizedBatch."""
        max_len = max(item.seq_length for item in items)
        batch_size = len(items)
        n_numeric = len(NUMERIC_FIELDS)

        pitch_type_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        pitch_result_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        bb_type_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        stand_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        p_throws_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        pa_event_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        numeric_features = torch.zeros(batch_size, max_len, n_numeric, dtype=torch.float32)
        numeric_mask = torch.zeros(batch_size, max_len, n_numeric, dtype=torch.bool)
        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        player_token_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        game_ids = torch.full((batch_size, max_len), PAD_GAME_ID, dtype=torch.long)
        seq_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, item in enumerate(items):
            sl = item.seq_length
            pitch_type_ids[i, :sl] = item.pitch_type_ids
            pitch_result_ids[i, :sl] = item.pitch_result_ids
            bb_type_ids[i, :sl] = item.bb_type_ids
            stand_ids[i, :sl] = item.stand_ids
            p_throws_ids[i, :sl] = item.p_throws_ids
            pa_event_ids[i, :sl] = item.pa_event_ids
            numeric_features[i, :sl] = item.numeric_features
            numeric_mask[i, :sl] = item.numeric_mask
            padding_mask[i, :sl] = item.padding_mask
            player_token_mask[i, :sl] = item.player_token_mask
            game_ids[i, :sl] = item.game_ids
            seq_lengths[i] = sl

        return TensorizedBatch(
            pitch_type_ids=pitch_type_ids,
            pitch_result_ids=pitch_result_ids,
            bb_type_ids=bb_type_ids,
            stand_ids=stand_ids,
            p_throws_ids=p_throws_ids,
            pa_event_ids=pa_event_ids,
            numeric_features=numeric_features,
            numeric_mask=numeric_mask,
            padding_mask=padding_mask,
            player_token_mask=player_token_mask,
            game_ids=game_ids,
            seq_lengths=seq_lengths,
        )

    def _encode_numeric(self, pitch: PitchEvent) -> tuple[list[float], list[bool]]:
        """Extract numeric features and mask from a PitchEvent."""
        values: list[float] = []
        mask: list[bool] = []
        for field_name in NUMERIC_FIELDS:
            raw = getattr(pitch, field_name)
            if raw is None:
                values.append(0.0)
                mask.append(False)
            elif isinstance(raw, bool):
                values.append(1.0 if raw else 0.0)
                mask.append(True)
            else:
                values.append(float(raw))
                mask.append(True)
        return values, mask
