"""Test fixtures for Discord bot tests."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.discord.config import DiscordConfig


@pytest.fixture
def discord_config() -> DiscordConfig:
    """Create a test Discord configuration."""
    return DiscordConfig(
        bot_token="test-token-123",
        allowed_channels=(123456789, 987654321),
        stream_edit_interval=0.1,
    )


@pytest.fixture
def discord_config_no_channels() -> DiscordConfig:
    """Create a test Discord configuration without channel restrictions."""
    return DiscordConfig(
        bot_token="test-token-123",
        allowed_channels=None,
        stream_edit_interval=0.1,
    )
