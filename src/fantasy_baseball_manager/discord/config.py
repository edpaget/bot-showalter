"""Discord bot configuration.

Loads configuration from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DiscordConfig:
    """Configuration for the Discord bot.

    Attributes:
        bot_token: Discord bot token (required).
        allowed_channels: Optional tuple of channel IDs where the bot can respond.
            If None, bot responds in all channels.
        stream_edit_interval: Seconds between message edits during streaming (rate limit protection).
    """

    bot_token: str
    allowed_channels: tuple[int, ...] | None = None
    stream_edit_interval: float = 0.5


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""


def load_discord_config() -> DiscordConfig:
    """Load Discord configuration from environment variables.

    Environment variables:
        DISCORD_BOT_TOKEN: Required. The Discord bot token.
        DISCORD_ALLOWED_CHANNELS: Optional. Comma-separated list of channel IDs.

    Returns:
        DiscordConfig instance.

    Raises:
        ConfigurationError: If required environment variables are missing.
    """
    bot_token = os.environ.get("DISCORD_BOT_TOKEN")
    if not bot_token:
        raise ConfigurationError("DISCORD_BOT_TOKEN environment variable is required")

    allowed_channels: tuple[int, ...] | None = None
    allowed_channels_str = os.environ.get("DISCORD_ALLOWED_CHANNELS")
    if allowed_channels_str:
        try:
            allowed_channels = tuple(int(ch.strip()) for ch in allowed_channels_str.split(",") if ch.strip())
        except ValueError as e:
            raise ConfigurationError(f"DISCORD_ALLOWED_CHANNELS must be comma-separated integers: {e}") from e

    return DiscordConfig(
        bot_token=bot_token,
        allowed_channels=allowed_channels,
    )
