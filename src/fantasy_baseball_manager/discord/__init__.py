"""Discord bot frontend for the fantasy baseball agent.

This module provides a Discord bot that allows users to interact with the
fantasy baseball agent via @mentions in Discord threads. Each thread
maintains its own conversation history.

Public API:
    create_bot(config) -> FantasyBaseballBot
    run_bot(config) -> None (async)
    load_discord_config() -> DiscordConfig

Example usage:
    import asyncio
    from fantasy_baseball_manager.discord import load_discord_config, run_bot

    config = load_discord_config()
    asyncio.run(run_bot(config))
"""

from fantasy_baseball_manager.discord.bot import (
    FantasyBaseballBot,
    create_bot,
    run_bot,
)
from fantasy_baseball_manager.discord.config import (
    ConfigurationError,
    DiscordConfig,
    load_discord_config,
)

__all__ = [
    "ConfigurationError",
    "DiscordConfig",
    "FantasyBaseballBot",
    "create_bot",
    "load_discord_config",
    "run_bot",
]
