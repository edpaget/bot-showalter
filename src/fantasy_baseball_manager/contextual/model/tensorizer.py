"""Converts PitchEvent dataclasses into batched tensors for the model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.models import GameSequence, PitchEvent, PlayerContext
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

    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length from the model config."""
        return self._config.max_seq_len

    def tensorize_game(self, game: GameSequence) -> TensorizedSingle:
        """Tensorize a single game: [PLAYER] + pitches. No [CLS] token.

        All game_ids are set to 0 (sentinel); callers re-index during assembly.
        """
        n_numeric = len(NUMERIC_FIELDS)

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

        # [PLAYER] token
        pitch_type_ids.append(0)
        pitch_result_ids.append(0)
        bb_type_ids.append(0)
        stand_ids.append(0)
        p_throws_ids.append(0)
        pa_event_ids.append(0)
        numeric_features.append([0.0] * n_numeric)
        numeric_mask.append([False] * n_numeric)
        player_token_mask.append(True)
        game_id_list.append(0)

        # Encode each pitch
        for pitch in game.pitches:
            pitch_type_ids.append(self._pitch_type_vocab.encode(pitch.pitch_type))
            pitch_result_ids.append(self._pitch_result_vocab.encode(pitch.pitch_result))
            bb_type_ids.append(self._bb_type_vocab.encode(pitch.bb_type) if pitch.bb_type is not None else 0)
            stand_ids.append(self._handedness_vocab.encode(pitch.stand))
            p_throws_ids.append(self._handedness_vocab.encode(pitch.p_throws))
            pa_event_ids.append(self._pa_event_vocab.encode(pitch.pa_event) if pitch.pa_event is not None else 0)

            nums, mask = self._encode_numeric(pitch)
            numeric_features.append(nums)
            numeric_mask.append(mask)
            player_token_mask.append(False)
            game_id_list.append(0)

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

    def tensorize_context(self, context: PlayerContext) -> TensorizedSingle:
        """Convert a PlayerContext into a TensorizedSingle.

        Tensorizes each game independently, then assembles the window with
        a [CLS] prefix and re-indexed game IDs.
        """
        game_tensors = [self.tensorize_game(g) for g in context.games]
        return assemble_game_window(game_tensors, self._config.max_seq_len)

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


def assemble_game_window(
    games: list[TensorizedSingle],
    max_seq_len: int,
) -> TensorizedSingle:
    """Assemble pre-tensorized games into a context window with [CLS] prefix.

    Prepends a [CLS] token, concatenates game tensors, assigns game_ids 0..N-1,
    and truncates oldest games if total exceeds max_seq_len.
    """
    n_numeric = len(NUMERIC_FIELDS)

    # Compute total length and truncate oldest games if needed
    game_lengths = [g.seq_length for g in games]
    total = 1 + sum(game_lengths)  # +1 for CLS
    start = 0
    while total > max_seq_len and start < len(games):
        total -= game_lengths[start]
        start += 1
    games = games[start:]

    # CLS token arrays
    cls_pitch_type = torch.zeros(1, dtype=torch.long)
    cls_pitch_result = torch.zeros(1, dtype=torch.long)
    cls_bb_type = torch.zeros(1, dtype=torch.long)
    cls_stand = torch.zeros(1, dtype=torch.long)
    cls_p_throws = torch.zeros(1, dtype=torch.long)
    cls_pa_event = torch.zeros(1, dtype=torch.long)
    cls_numeric = torch.zeros(1, n_numeric, dtype=torch.float32)
    cls_numeric_mask = torch.zeros(1, n_numeric, dtype=torch.bool)
    cls_padding = torch.ones(1, dtype=torch.bool)
    cls_player_token = torch.zeros(1, dtype=torch.bool)
    cls_game_id = torch.tensor([CLS_GAME_ID], dtype=torch.long)

    # Concatenate game tensors with re-indexed game_ids
    all_pitch_type = [cls_pitch_type]
    all_pitch_result = [cls_pitch_result]
    all_bb_type = [cls_bb_type]
    all_stand = [cls_stand]
    all_p_throws = [cls_p_throws]
    all_pa_event = [cls_pa_event]
    all_numeric = [cls_numeric]
    all_numeric_mask = [cls_numeric_mask]
    all_padding = [cls_padding]
    all_player_token = [cls_player_token]
    all_game_id = [cls_game_id]

    for game_idx, g in enumerate(games):
        all_pitch_type.append(g.pitch_type_ids)
        all_pitch_result.append(g.pitch_result_ids)
        all_bb_type.append(g.bb_type_ids)
        all_stand.append(g.stand_ids)
        all_p_throws.append(g.p_throws_ids)
        all_pa_event.append(g.pa_event_ids)
        all_numeric.append(g.numeric_features)
        all_numeric_mask.append(g.numeric_mask)
        all_padding.append(g.padding_mask)
        all_player_token.append(g.player_token_mask)
        all_game_id.append(torch.full((g.seq_length,), game_idx, dtype=torch.long))

    seq_len = 1 + sum(g.seq_length for g in games)
    return TensorizedSingle(
        pitch_type_ids=torch.cat(all_pitch_type),
        pitch_result_ids=torch.cat(all_pitch_result),
        bb_type_ids=torch.cat(all_bb_type),
        stand_ids=torch.cat(all_stand),
        p_throws_ids=torch.cat(all_p_throws),
        pa_event_ids=torch.cat(all_pa_event),
        numeric_features=torch.cat(all_numeric),
        numeric_mask=torch.cat(all_numeric_mask),
        padding_mask=torch.cat(all_padding),
        player_token_mask=torch.cat(all_player_token),
        game_ids=torch.cat(all_game_id),
        seq_length=seq_len,
    )
