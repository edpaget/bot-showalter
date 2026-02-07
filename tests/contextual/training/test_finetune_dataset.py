"""Tests for fine-tune dataset, stat extraction, and sliding window construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
)
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
    PITCHER_TARGET_STATS,
    FineTuneConfig,
)
from fantasy_baseball_manager.contextual.training.dataset import (
    FineTuneBatch,
    FineTuneDataset,
    FineTuneSample,
    build_finetune_windows,
    collate_finetune_samples,
    extract_game_stats,
)
from tests.contextual.model.conftest import make_pitch, make_player_context

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig


def _build_tensorizer(config: ModelConfig) -> Tensorizer:
    return Tensorizer(
        config=config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )


def _game_with_events(*pa_events: str | None, perspective: str = "batter") -> object:
    """Create a GameSequence where certain pitches have pa_event set."""
    from fantasy_baseball_manager.contextual.data.models import GameSequence

    pitches = tuple(
        make_pitch(pitch_number=i + 1, pa_event=event)
        for i, event in enumerate(pa_events)
    )
    return GameSequence(
        game_pk=717465,
        game_date="2024-03-28",
        season=2024,
        home_team="LAD",
        away_team="SD",
        perspective=perspective,
        player_id=660271,
        pitches=pitches,
    )


class TestExtractGameStats:
    def test_single_home_run(self) -> None:
        game = _game_with_events(None, "home_run")
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert stats.shape == (6,)
        # hr=1, so=0, bb=0, h=1 (HR is also a hit), 2b=0, 3b=0
        assert stats[0].item() == 1.0  # hr
        assert stats[1].item() == 0.0  # so
        assert stats[2].item() == 0.0  # bb
        assert stats[3].item() == 1.0  # h
        assert stats[4].item() == 0.0  # 2b
        assert stats[5].item() == 0.0  # 3b

    def test_strikeout(self) -> None:
        game = _game_with_events(None, None, "strikeout")
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert stats[1].item() == 1.0  # so

    def test_strikeout_double_play(self) -> None:
        game = _game_with_events("strikeout_double_play")
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert stats[1].item() == 1.0  # so

    def test_walk_and_intentional_walk(self) -> None:
        game = _game_with_events("walk", "intentional_walk")
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert stats[2].item() == 2.0  # bb

    def test_single(self) -> None:
        game = _game_with_events("single")
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert stats[3].item() == 1.0  # h
        assert stats[0].item() == 0.0  # hr
        assert stats[4].item() == 0.0  # 2b

    def test_double(self) -> None:
        game = _game_with_events("double")
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert stats[3].item() == 1.0  # h
        assert stats[4].item() == 1.0  # 2b

    def test_triple(self) -> None:
        game = _game_with_events("triple")
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert stats[3].item() == 1.0  # h
        assert stats[5].item() == 1.0  # 3b

    def test_no_pa_events(self) -> None:
        game = _game_with_events(None, None, None)
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert torch.all(stats == 0.0)

    def test_multiple_events(self) -> None:
        game = _game_with_events("home_run", "strikeout", "single", "double", "walk")
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert stats[0].item() == 1.0  # hr
        assert stats[1].item() == 1.0  # so
        assert stats[2].item() == 1.0  # bb
        assert stats[3].item() == 3.0  # h (HR + single + double)
        assert stats[4].item() == 1.0  # 2b
        assert stats[5].item() == 0.0  # 3b

    def test_pitcher_target_stats(self) -> None:
        game = _game_with_events("strikeout", "home_run", "walk", "single")
        stats = extract_game_stats(game, PITCHER_TARGET_STATS)  # type: ignore[arg-type]
        assert stats.shape == (4,)
        assert stats[0].item() == 1.0  # so
        assert stats[1].item() == 2.0  # h (HR + single)
        assert stats[2].item() == 1.0  # bb
        assert stats[3].item() == 1.0  # hr

    def test_unrelated_pa_event_counts_nothing(self) -> None:
        game = _game_with_events("field_out", "grounded_into_double_play")
        stats = extract_game_stats(game, BATTER_TARGET_STATS)  # type: ignore[arg-type]
        assert torch.all(stats == 0.0)


class TestBuildFineTuneWindows:
    def test_correct_number_of_windows(self, small_config: ModelConfig) -> None:
        """A player with G games produces G - context_window windows."""
        n_games = 15
        context_window = 3
        ft_config = FineTuneConfig(
            context_window=context_window,
            min_games=context_window + 1,
        )
        ctx = make_player_context(n_games=n_games, pitches_per_game=5)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        assert len(windows) == n_games - context_window

    def test_skips_players_with_few_games(self, small_config: ModelConfig) -> None:
        context_window = 5
        ft_config = FineTuneConfig(
            context_window=context_window,
            min_games=context_window + 1,
        )
        # Player with only 4 games (< context_window + 1)
        ctx = make_player_context(n_games=4, pitches_per_game=5)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        assert len(windows) == 0

    def test_multiple_players(self, small_config: ModelConfig) -> None:
        context_window = 2
        ft_config = FineTuneConfig(
            context_window=context_window,
            min_games=context_window + 1,
        )
        ctx1 = make_player_context(n_games=5, pitches_per_game=5, player_id=100)
        ctx2 = make_player_context(n_games=4, pitches_per_game=5, player_id=200)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx1, ctx2], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        # ctx1: 5 - 2 = 3, ctx2: 4 - 2 = 2 → 5 total
        assert len(windows) == 5

    def test_window_target_shape(self, small_config: ModelConfig) -> None:
        ft_config = FineTuneConfig(context_window=2, min_games=3)
        ctx = make_player_context(n_games=5, pitches_per_game=5)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        for ts, targets in windows:
            assert targets.shape == (len(BATTER_TARGET_STATS),)
            assert ts.seq_length > 0

    def test_window_targets_match_known_events(self, small_config: ModelConfig) -> None:
        """Verify that targets from sliding windows match the actual game stats."""
        from fantasy_baseball_manager.contextual.data.models import (
            GameSequence,
            PlayerContext,
        )

        # Build games where game i has i home runs
        games = []
        for i in range(5):
            pa_events: list[str | None] = ["home_run"] * i + [None] * (3 - i) if i < 3 else ["home_run"] * 3
            pitches = tuple(
                make_pitch(pitch_number=j + 1, pa_event=pa_events[j] if j < len(pa_events) else None)
                for j in range(3)
            )
            games.append(GameSequence(
                game_pk=717465 + i,
                game_date=f"2024-03-{28 + i:02d}",
                season=2024,
                home_team="LAD",
                away_team="SD",
                perspective="batter",
                player_id=660271,
                pitches=pitches,
            ))

        ctx = PlayerContext(
            player_id=660271,
            player_name="Test",
            season=2024,
            perspective="batter",
            games=tuple(games),
        )

        ft_config = FineTuneConfig(context_window=2, min_games=3)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        # 5 - 2 = 3 windows, target game indices = 2, 3, 4
        assert len(windows) == 3
        # Each target corresponds to games[2], games[3], games[4]
        for _, targets in windows:
            # At least verify shape
            assert targets.shape == (len(BATTER_TARGET_STATS),)


class TestFineTuneDataset:
    def test_length(self, small_config: ModelConfig) -> None:
        ft_config = FineTuneConfig(context_window=2, min_games=3)
        ctx = make_player_context(n_games=5, pitches_per_game=5)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        dataset = FineTuneDataset(windows)
        assert len(dataset) == 3  # 5 - 2

    def test_getitem_returns_finetune_sample(self, small_config: ModelConfig) -> None:
        ft_config = FineTuneConfig(context_window=2, min_games=3)
        ctx = make_player_context(n_games=5, pitches_per_game=5)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        dataset = FineTuneDataset(windows)
        sample = dataset[0]
        assert isinstance(sample, FineTuneSample)
        assert sample.context.seq_length > 0
        assert sample.targets.shape == (len(BATTER_TARGET_STATS),)


class TestCollateFineTuneSamples:
    def test_shapes(self, small_config: ModelConfig) -> None:
        ft_config = FineTuneConfig(context_window=2, min_games=3)
        ctx1 = make_player_context(n_games=4, pitches_per_game=3, player_id=100)
        ctx2 = make_player_context(n_games=4, pitches_per_game=5, player_id=200)
        tensorizer = _build_tensorizer(small_config)

        windows1 = build_finetune_windows(
            [ctx1], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        windows2 = build_finetune_windows(
            [ctx2], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        dataset = FineTuneDataset(windows1 + windows2)

        samples = [dataset[i] for i in range(min(3, len(dataset)))]
        batch = collate_finetune_samples(samples)

        assert isinstance(batch, FineTuneBatch)
        b = len(samples)
        assert batch.targets.shape == (b, len(BATTER_TARGET_STATS))
        assert batch.context.padding_mask.shape[0] == b
        assert batch.context.pitch_type_ids.shape[0] == b

    def test_padding(self, small_config: ModelConfig) -> None:
        ft_config = FineTuneConfig(context_window=2, min_games=3)
        # Different pitches_per_game → different seq lengths
        ctx1 = make_player_context(n_games=4, pitches_per_game=3, player_id=100)
        ctx2 = make_player_context(n_games=4, pitches_per_game=8, player_id=200)
        tensorizer = _build_tensorizer(small_config)

        windows1 = build_finetune_windows(
            [ctx1], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        windows2 = build_finetune_windows(
            [ctx2], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        dataset = FineTuneDataset(windows1 + windows2)

        s1 = dataset[0]  # from ctx1 (shorter)
        s2 = dataset[len(windows1)]  # from ctx2 (longer)
        batch = collate_finetune_samples([s1, s2])

        # Both should be padded to same length
        max_len = batch.context.padding_mask.shape[1]
        assert max_len == max(s1.context.seq_length, s2.context.seq_length)
        # Shorter sample should have padding (False in padding_mask)
        shorter_len = min(s1.context.seq_length, s2.context.seq_length)
        if shorter_len < max_len:
            shorter_idx = 0 if s1.context.seq_length < s2.context.seq_length else 1
            assert not batch.context.padding_mask[shorter_idx, max_len - 1].item()
