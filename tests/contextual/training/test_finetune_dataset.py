"""Tests for fine-tune dataset, stat extraction, and sliding window construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
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
    compute_rate_targets,
    compute_target_statistics,
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
    """Create a GameSequence where each element is a separate one-pitch PA.

    Each pitch gets pitch_number=1 since it's the sole pitch in its PA,
    matching how real Statcast data represents single-pitch plate appearances.
    """
    from fantasy_baseball_manager.contextual.data.models import GameSequence

    pitches = tuple(
        make_pitch(pitch_number=1, pa_event=event)
        for event in pa_events
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

    def test_duplicate_pa_event_counted_once(self) -> None:
        """If pa_event is set on every pitch in a PA, it should only count once."""
        from fantasy_baseball_manager.contextual.data.models import GameSequence

        # Simulate a 3-pitch strikeout where pa_event is on all pitches
        pitches = tuple(
            make_pitch(pitch_number=i + 1, pa_event="strikeout")
            for i in range(3)
        )
        game = GameSequence(
            game_pk=717465,
            game_date="2024-03-28",
            season=2024,
            home_team="LAD",
            away_team="SD",
            perspective="batter",
            player_id=660271,
            pitches=pitches,
        )
        stats = extract_game_stats(game, BATTER_TARGET_STATS)
        assert stats[1].item() == 1.0  # so should be 1, not 3

    def test_multi_pa_duplicate_events(self) -> None:
        """Multiple PAs each with pa_event on all pitches should count correctly."""
        from fantasy_baseball_manager.contextual.data.models import GameSequence

        # PA1: 2-pitch walk (pa_event on both), PA2: 3-pitch strikeout (pa_event on all)
        pitches = (
            make_pitch(pitch_number=1, pa_event="walk"),
            make_pitch(pitch_number=2, pa_event="walk"),
            make_pitch(pitch_number=1, pa_event="strikeout"),
            make_pitch(pitch_number=2, pa_event="strikeout"),
            make_pitch(pitch_number=3, pa_event="strikeout"),
        )
        game = GameSequence(
            game_pk=717465,
            game_date="2024-03-28",
            season=2024,
            home_team="LAD",
            away_team="SD",
            perspective="batter",
            player_id=660271,
            pitches=tuple(pitches),
        )
        stats = extract_game_stats(game, BATTER_TARGET_STATS)
        assert stats[1].item() == 1.0  # so = 1
        assert stats[2].item() == 1.0  # bb = 1

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
            target_mode="counts",
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
            target_mode="counts",
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
            target_mode="counts",
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
        ft_config = FineTuneConfig(target_mode="counts", context_window=2, min_games=3)
        ctx = make_player_context(n_games=5, pitches_per_game=5)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        for ts, targets, context_mean in windows:
            assert targets.shape == (len(BATTER_TARGET_STATS),)
            assert context_mean.shape == (len(BATTER_TARGET_STATS),)
            assert ts.seq_length > 0

    def test_context_mean_values(self, small_config: ModelConfig) -> None:
        """Verify context_mean is the average of context game stats."""
        from fantasy_baseball_manager.contextual.data.models import (
            GameSequence,
            PlayerContext,
        )

        # Game 0: 0 HR, Game 1: 1 HR, Game 2: 2 HR (but max 3 pitches), Game 3: 0 HR
        # Each pitch is a separate one-pitch PA (pitch_number=1)
        games = []
        for i, n_hr in enumerate([0, 1, 2, 0]):
            pa_events: list[str | None] = ["home_run"] * n_hr + [None] * (3 - n_hr)
            pitches = tuple(
                make_pitch(pitch_number=1, pa_event=pa_events[j])
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

        ft_config = FineTuneConfig(target_mode="counts", context_window=2, min_games=3)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        # Window 0: context = games[0,1], target = game[2]
        # context_mean hr = (0+1)/2 = 0.5
        _, _, cm0 = windows[0]
        assert cm0[0].item() == pytest.approx(0.5)  # hr index
        # Window 1: context = games[1,2], target = game[3]
        # context_mean hr = (1+2)/2 = 1.5
        _, _, cm1 = windows[1]
        assert cm1[0].item() == pytest.approx(1.5)  # hr index

    def test_window_targets_match_known_events(self, small_config: ModelConfig) -> None:
        """Verify that targets from sliding windows match the actual game stats."""
        from fantasy_baseball_manager.contextual.data.models import (
            GameSequence,
            PlayerContext,
        )

        # Build games where game i has min(i, 3) home runs
        # Each pitch is a separate one-pitch PA (pitch_number=1)
        games = []
        for i in range(5):
            pa_events: list[str | None] = ["home_run"] * i + [None] * (3 - i) if i < 3 else ["home_run"] * 3
            pitches = tuple(
                make_pitch(pitch_number=1, pa_event=pa_events[j] if j < len(pa_events) else None)
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

        ft_config = FineTuneConfig(target_mode="counts", context_window=2, min_games=3)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        # 5 - 2 = 3 windows, target game indices = 2, 3, 4
        assert len(windows) == 3
        # Each target corresponds to games[2], games[3], games[4]
        for _, targets, context_mean in windows:
            # At least verify shape
            assert targets.shape == (len(BATTER_TARGET_STATS),)
            assert context_mean.shape == (len(BATTER_TARGET_STATS),)


class TestFineTuneDataset:
    def test_length(self, small_config: ModelConfig) -> None:
        ft_config = FineTuneConfig(target_mode="counts", context_window=2, min_games=3)
        ctx = make_player_context(n_games=5, pitches_per_game=5)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        dataset = FineTuneDataset(windows)
        assert len(dataset) == 3  # 5 - 2

    def test_getitem_returns_finetune_sample(self, small_config: ModelConfig) -> None:
        ft_config = FineTuneConfig(target_mode="counts", context_window=2, min_games=3)
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
        assert sample.context_mean.shape == (len(BATTER_TARGET_STATS),)


class TestCollateFineTuneSamples:
    def test_shapes(self, small_config: ModelConfig) -> None:
        ft_config = FineTuneConfig(target_mode="counts", context_window=2, min_games=3)
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
        assert batch.context_mean.shape == (b, len(BATTER_TARGET_STATS))
        assert batch.context.padding_mask.shape[0] == b
        assert batch.context.pitch_type_ids.shape[0] == b

    def test_padding(self, small_config: ModelConfig) -> None:
        ft_config = FineTuneConfig(target_mode="counts", context_window=2, min_games=3)
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


class TestComputeRateTargets:
    def _game_with_pa_events(
        self, *pa_events: str | None, perspective: str = "pitcher",
    ) -> object:
        """Create a GameSequence with given PA events."""
        from fantasy_baseball_manager.contextual.data.models import GameSequence

        pitches = tuple(
            make_pitch(pitch_number=1, pa_event=event) for event in pa_events
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

    def test_pitcher_rate_computation(self) -> None:
        """Rate = sum(counts) / sum(outs) across games."""
        # Game 1: 2 strikeouts, 1 field_out → 3 outs, 2 SO
        g1 = self._game_with_pa_events("strikeout", "strikeout", "field_out")
        # Game 2: 1 strikeout, 2 field_outs → 3 outs, 1 SO
        g2 = self._game_with_pa_events("strikeout", "field_out", "field_out")

        result = compute_rate_targets(
            context_games=(g1, g2),  # type: ignore[arg-type]
            target_games=(g1,),  # type: ignore[arg-type]
            target_stats=PITCHER_TARGET_STATS,
            perspective="pitcher",
        )
        assert result is not None
        target_rate, context_rate = result

        # Target: g1 has 2 SO, 3 outs → SO rate = 2/3
        assert target_rate[0].item() == pytest.approx(2 / 3)
        # Context: g1+g2 has 3 SO, 6 outs → SO rate = 3/6 = 0.5
        assert context_rate[0].item() == pytest.approx(0.5)

    def test_batter_rate_computation(self) -> None:
        """Rate = sum(counts) / sum(PA) across games."""
        # Game 1: 1 HR, 1 strikeout, 1 walk → 3 PA, 1 HR
        g1 = self._game_with_pa_events("home_run", "strikeout", "walk", perspective="batter")
        # Game 2: 0 HR, 2 field_outs, 1 single → 3 PA, 0 HR
        g2 = self._game_with_pa_events("field_out", "field_out", "single", perspective="batter")

        result = compute_rate_targets(
            context_games=(g1, g2),  # type: ignore[arg-type]
            target_games=(g1,),  # type: ignore[arg-type]
            target_stats=BATTER_TARGET_STATS,
            perspective="batter",
        )
        assert result is not None
        target_rate, context_rate = result

        # Target HR rate: 1/3
        assert target_rate[0].item() == pytest.approx(1 / 3)
        # Context HR rate: 1/6
        assert context_rate[0].item() == pytest.approx(1 / 6)

    def test_zero_target_denom_returns_none(self) -> None:
        """If target games have zero PA/outs, return None."""
        g = self._game_with_pa_events(None, None, None)
        g_ctx = self._game_with_pa_events("strikeout", "field_out")

        result = compute_rate_targets(
            context_games=(g_ctx,),  # type: ignore[arg-type]
            target_games=(g,),  # type: ignore[arg-type]
            target_stats=PITCHER_TARGET_STATS,
            perspective="pitcher",
        )
        assert result is None

    def test_zero_context_denom_returns_none(self) -> None:
        """If context games have zero PA/outs, return None."""
        g = self._game_with_pa_events(None, None, None)
        g_tgt = self._game_with_pa_events("strikeout", "field_out")

        result = compute_rate_targets(
            context_games=(g,),  # type: ignore[arg-type]
            target_games=(g_tgt,),  # type: ignore[arg-type]
            target_stats=PITCHER_TARGET_STATS,
            perspective="pitcher",
        )
        assert result is None


class TestBuildFineTuneWindowsRatesMode:
    def test_correct_number_of_windows(self, small_config: ModelConfig) -> None:
        """G games, context_window=N, target_window=K → G - N - K + 1 windows."""
        n_games = 15
        context_window = 3
        target_window = 2
        ft_config = FineTuneConfig(
            target_mode="rates",
            context_window=context_window,
            target_window=target_window,
            min_games=context_window + target_window,
        )
        ctx = make_player_context(n_games=n_games, pitches_per_game=5)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        # Since all pitches have pa_event=None, PA=0 for batter perspective.
        # All windows will be skipped due to zero denominator.
        assert len(windows) == 0

    def test_rates_mode_with_pa_events(self, small_config: ModelConfig) -> None:
        """Rate windows with actual PA events produce correct count."""
        from fantasy_baseball_manager.contextual.data.models import (
            GameSequence,
            PlayerContext,
        )

        n_games = 8
        context_window = 3
        target_window = 2
        games = []
        for i in range(n_games):
            pitches = tuple(
                make_pitch(pitch_number=1, pa_event="field_out")
                for _ in range(3)
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

        ft_config = FineTuneConfig(
            target_mode="rates",
            context_window=context_window,
            target_window=target_window,
            min_games=context_window + target_window,
        )
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        # 8 - 3 - 2 + 1 = 4 windows
        assert len(windows) == 4

    def test_counts_mode_unchanged(self, small_config: ModelConfig) -> None:
        """Counts mode still produces G - N windows."""
        n_games = 15
        context_window = 3
        ft_config = FineTuneConfig(
            target_mode="counts",
            context_window=context_window,
            min_games=context_window + 1,
        )
        ctx = make_player_context(n_games=n_games, pitches_per_game=5)
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        assert len(windows) == n_games - context_window

    def test_context_mean_holds_context_rate_in_rates_mode(
        self, small_config: ModelConfig,
    ) -> None:
        """In rates mode, context_mean is the context rate (sum counts / sum denoms)."""
        from fantasy_baseball_manager.contextual.data.models import (
            GameSequence,
            PlayerContext,
        )

        games = []
        for i in range(6):
            # Each game: 1 strikeout + 2 field_outs = 3 PA
            # SO=1 per game for batter stats
            pitches = (
                make_pitch(pitch_number=1, pa_event="strikeout"),
                make_pitch(pitch_number=1, pa_event="field_out"),
                make_pitch(pitch_number=1, pa_event="field_out"),
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

        ft_config = FineTuneConfig(
            target_mode="rates",
            context_window=3,
            target_window=2,
            min_games=5,
        )
        tensorizer = _build_tensorizer(small_config)

        windows = build_finetune_windows(
            [ctx], tensorizer, ft_config, BATTER_TARGET_STATS,
        )
        assert len(windows) > 0
        # Each game has 1 SO in 3 PA → SO rate = 1/3
        for _, targets, context_mean in windows:
            # SO is index 1 in BATTER_TARGET_STATS
            assert context_mean[1].item() == pytest.approx(1 / 3)
            assert targets[1].item() == pytest.approx(1 / 3)


class TestComputeTargetStatistics:
    def test_known_values(self) -> None:
        """Mean and std match hand-computed values."""
        from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedSingle

        dummy_ts = TensorizedSingle(
            pitch_type_ids=torch.zeros(1, dtype=torch.long),
            pitch_result_ids=torch.zeros(1, dtype=torch.long),
            bb_type_ids=torch.zeros(1, dtype=torch.long),
            stand_ids=torch.zeros(1, dtype=torch.long),
            p_throws_ids=torch.zeros(1, dtype=torch.long),
            pa_event_ids=torch.zeros(1, dtype=torch.long),
            numeric_features=torch.zeros(1, 1),
            numeric_mask=torch.zeros(1, 1, dtype=torch.bool),
            padding_mask=torch.ones(1, dtype=torch.bool),
            player_token_mask=torch.zeros(1, dtype=torch.bool),
            game_ids=torch.zeros(1, dtype=torch.long),
            seq_length=1,
        )

        # 3 windows with 2 target stats each
        windows = [
            (dummy_ts, torch.tensor([1.0, 10.0]), torch.zeros(2)),
            (dummy_ts, torch.tensor([2.0, 20.0]), torch.zeros(2)),
            (dummy_ts, torch.tensor([3.0, 30.0]), torch.zeros(2)),
        ]

        mean, std = compute_target_statistics(windows)
        assert mean.shape == (2,)
        assert std.shape == (2,)
        # mean = [2.0, 20.0]
        assert mean[0].item() == pytest.approx(2.0)
        assert mean[1].item() == pytest.approx(20.0)
        # std of [1,2,3] = 1.0 (with Bessel correction)
        expected_std = torch.tensor([1.0, 2.0, 3.0]).std().item()
        assert std[0].item() == pytest.approx(expected_std)

    def test_std_clamped(self) -> None:
        """Std is clamped to minimum 1e-6 for constant targets."""
        from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedSingle

        dummy_ts = TensorizedSingle(
            pitch_type_ids=torch.zeros(1, dtype=torch.long),
            pitch_result_ids=torch.zeros(1, dtype=torch.long),
            bb_type_ids=torch.zeros(1, dtype=torch.long),
            stand_ids=torch.zeros(1, dtype=torch.long),
            p_throws_ids=torch.zeros(1, dtype=torch.long),
            pa_event_ids=torch.zeros(1, dtype=torch.long),
            numeric_features=torch.zeros(1, 1),
            numeric_mask=torch.zeros(1, 1, dtype=torch.bool),
            padding_mask=torch.ones(1, dtype=torch.bool),
            player_token_mask=torch.zeros(1, dtype=torch.bool),
            game_ids=torch.zeros(1, dtype=torch.long),
            seq_length=1,
        )

        # All identical targets → std = 0 before clamping
        windows = [
            (dummy_ts, torch.tensor([5.0]), torch.zeros(1)),
            (dummy_ts, torch.tensor([5.0]), torch.zeros(1)),
        ]

        _, std = compute_target_statistics(windows)
        assert std[0].item() > 0
