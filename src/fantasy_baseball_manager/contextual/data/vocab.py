"""Vocabularies for categorical pitch event fields.

Each vocabulary maps string tokens to integer indices for use in embedding
layers. All vocabularies reserve index 0 for <PAD> and index 1 for <UNK>.
"""

from __future__ import annotations

from dataclasses import dataclass


def _build_vocab(name: str, tokens: list[str]) -> Vocabulary:
    """Build a Vocabulary with <PAD>=0, <UNK>=1, then tokens starting at 2."""
    token_to_index = {"<PAD>": 0, "<UNK>": 1}
    for i, token in enumerate(tokens, start=2):
        token_to_index[token] = i
    return Vocabulary(name=name, token_to_index=token_to_index)


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Maps string tokens to integer indices with <UNK> fallback.

    Attributes:
        name: Human-readable name for this vocabulary.
        token_to_index: Mapping from token strings to integer indices.
    """

    name: str
    token_to_index: dict[str, int]

    def encode(self, token: str) -> int:
        """Encode a token to its integer index, falling back to <UNK>."""
        return self.token_to_index.get(token, self.token_to_index["<UNK>"])

    @property
    def size(self) -> int:
        """Number of tokens in the vocabulary (including PAD and UNK)."""
        return len(self.token_to_index)

    @property
    def index_to_token(self) -> dict[int, str]:
        """Reverse mapping from integer indices to token strings."""
        return {v: k for k, v in self.token_to_index.items()}


PITCH_TYPE_VOCAB: Vocabulary = _build_vocab(
    "pitch_type",
    [
        "FF",  # 4-Seam Fastball
        "SI",  # Sinker
        "FC",  # Cutter
        "SL",  # Slider
        "ST",  # Sweeper
        "SV",  # Slurve
        "CU",  # Curveball
        "KC",  # Knuckle Curve
        "CS",  # Slow Curve
        "CH",  # Changeup
        "FS",  # Splitter
        "KN",  # Knuckleball
        "EP",  # Eephus
        "SC",  # Screwball
        "FA",  # Fastball (generic)
        "FO",  # Forkball
        "IN",  # Intentional Ball
        "PO",  # Pitchout
        "AB",  # Automatic Ball
    ],
)

PITCH_RESULT_VOCAB: Vocabulary = _build_vocab(
    "pitch_result",
    [
        "ball",
        "called_strike",
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_tip",
        "foul_bunt",
        "missed_bunt",
        "bunt_foul_tip",
        "hit_into_play",
        "hit_into_play_no_out",
        "hit_into_play_score",
        "hit_by_pitch",
        "blocked_ball",
        "pitchout",
    ],
)

PA_EVENT_VOCAB: Vocabulary = _build_vocab(
    "pa_event",
    [
        "single",
        "double",
        "triple",
        "home_run",
        "strikeout",
        "strikeout_double_play",
        "walk",
        "intentional_walk",
        "hit_by_pitch",
        "field_out",
        "grounded_into_double_play",
        "double_play",
        "force_out",
        "fielders_choice",
        "fielders_choice_out",
        "field_error",
        "sac_fly",
        "sac_fly_double_play",
        "sac_bunt",
        "sac_bunt_double_play",
        "triple_play",
        "catcher_interf",
        "fan_interference",
        "batter_interference",
        "other_out",
    ],
)

BB_TYPE_VOCAB: Vocabulary = _build_vocab(
    "bb_type",
    [
        "ground_ball",
        "line_drive",
        "fly_ball",
        "popup",
    ],
)

HANDEDNESS_VOCAB: Vocabulary = _build_vocab(
    "handedness",
    [
        "R",
        "L",
    ],
)
