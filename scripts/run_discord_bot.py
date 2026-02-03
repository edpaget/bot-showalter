#!/usr/bin/env python
"""Entry point script for running the Discord bot.

Usage:
    DISCORD_BOT_TOKEN=xxx uv run python scripts/run_discord_bot.py

Environment variables:
    DISCORD_BOT_TOKEN: Required. The Discord bot token.
    DISCORD_ALLOWED_CHANNELS: Optional. Comma-separated list of allowed channel IDs.
    ANTHROPIC_API_KEY: Required. API key for the Anthropic LLM.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from fantasy_baseball_manager.discord import ConfigurationError, load_discord_config, run_bot


def main() -> int:
    """Run the Discord bot."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        config = load_discord_config()
    except ConfigurationError as e:
        logger.error("Configuration error: %s", e)
        return 1

    logger.info("Starting Discord bot...")
    try:
        asyncio.run(run_bot(config))
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception:
        logger.exception("Bot crashed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
