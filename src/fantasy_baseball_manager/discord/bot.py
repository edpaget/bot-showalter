"""Discord bot client for the fantasy baseball agent.

Provides a Discord client that responds to @mentions, streaming responses
from the fantasy baseball agent.

- Thread mentions: Maintains conversation history via cached agents per thread
- Main channel mentions: Responds with a fresh agent (no history)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import discord
from fantasy_baseball_manager.agent import create_agent, stream

if TYPE_CHECKING:
    from fantasy_baseball_manager.agent import Agent
    from fantasy_baseball_manager.discord.config import DiscordConfig

logger = logging.getLogger(__name__)


class FantasyBaseballBot(discord.Client):
    """Discord bot that interfaces with the fantasy baseball agent.

    - Thread mentions: Each thread maintains its own conversation history via
      cached thread-specific agents.
    - Main channel mentions: Responds with a fresh agent per message (no history).
    """

    def __init__(self, config: DiscordConfig, **kwargs) -> None:
        """Initialize the bot.

        Args:
            config: Discord configuration.
            **kwargs: Additional arguments passed to discord.Client.
        """
        # Set up intents - we need message content to read mentions
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(intents=intents, **kwargs)

        self._config = config
        self._agents: dict[str, Agent] = {}
        self._lock = asyncio.Lock()

    async def on_ready(self) -> None:
        """Handle bot ready event."""
        logger.info("Bot is ready. Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages.

        Responds to @mentions in both threads and main channels:
        - Threads: Uses cached agents with conversation history
        - Main channels: Uses ephemeral agents (no history)

        Args:
            message: The incoming Discord message.
        """
        # Ignore messages from the bot itself
        if message.author == self.user:
            return

        # Check if bot was mentioned
        if not self.user or self.user not in message.mentions:
            return

        # Check allowed channels if configured
        if self._config.allowed_channels:
            if isinstance(message.channel, discord.Thread):
                channel_id = message.channel.parent_id
            else:
                channel_id = message.channel.id
            if channel_id not in self._config.allowed_channels:
                logger.debug("Ignoring message in non-allowed channel: %s", channel_id)
                return

        # Extract message content (remove bot mention)
        content = self._extract_message_content(message)
        if not content:
            await message.reply("Please include a message with your mention!", mention_author=False)
            return

        # Get or create agent based on context
        if isinstance(message.channel, discord.Thread):
            # Thread: use cached agent for conversation history
            context_id = str(message.channel.id)
            agent = await self._get_or_create_agent(context_id)
        else:
            # Main channel: ephemeral agent (no caching, no history)
            context_id = str(message.id)
            agent = create_agent(thread_id=context_id)

        # Stream the response
        await self._stream_response(message, agent, content)

    def _extract_message_content(self, message: discord.Message) -> str:
        """Extract the user's message content, removing bot mentions.

        Args:
            message: The Discord message.

        Returns:
            The message content with bot mentions removed.
        """
        content = message.content

        # Remove bot mention patterns
        if self.user:
            # Remove <@!ID> and <@ID> patterns for the bot
            content = content.replace(f"<@!{self.user.id}>", "")
            content = content.replace(f"<@{self.user.id}>", "")

        return content.strip()

    async def _get_or_create_agent(self, thread_id: str) -> Agent:
        """Get or create an agent for the given thread.

        Thread-safe agent cache access.

        Args:
            thread_id: The Discord thread ID.

        Returns:
            The agent for this thread.
        """
        async with self._lock:
            if thread_id not in self._agents:
                logger.info("Creating new agent for thread %s", thread_id)
                self._agents[thread_id] = create_agent(thread_id=thread_id)
            return self._agents[thread_id]

    async def _stream_response(self, message: discord.Message, agent: Agent, content: str) -> None:
        """Stream a response from the agent, editing the message periodically.

        Args:
            message: The original Discord message to reply to.
            agent: The agent to use for generating the response.
            content: The user's message content.
        """
        # Send initial "Thinking..." message
        response_msg = await message.reply("Thinking...", mention_author=False)

        accumulated = ""
        last_edit_time = asyncio.get_event_loop().time()

        try:
            async for chunk in stream(agent, content):
                accumulated += chunk

                # Rate-limit message edits
                current_time = asyncio.get_event_loop().time()
                if current_time - last_edit_time >= self._config.stream_edit_interval:
                    await response_msg.edit(content=self._truncate_response(accumulated))
                    last_edit_time = current_time

            # Final edit with complete response
            if accumulated:
                await response_msg.edit(content=self._truncate_response(accumulated))
            else:
                await response_msg.edit(content="I couldn't generate a response. Please try again.")

        except Exception:
            logger.exception("Error streaming response for thread %s", message.channel.id)
            await response_msg.edit(content="An error occurred while processing your request. Please try again.")

    def _truncate_response(self, content: str, max_length: int = 2000) -> str:
        """Truncate response to fit Discord's message limit.

        Args:
            content: The response content.
            max_length: Maximum message length (Discord limit is 2000).

        Returns:
            Truncated content if necessary.
        """
        if len(content) <= max_length:
            return content
        return content[: max_length - 3] + "..."


def create_bot(config: DiscordConfig) -> FantasyBaseballBot:
    """Create a new Discord bot instance.

    Args:
        config: Discord configuration.

    Returns:
        A configured FantasyBaseballBot instance.
    """
    return FantasyBaseballBot(config)


async def run_bot(config: DiscordConfig) -> None:
    """Run the Discord bot.

    This is an async function that runs the bot until it disconnects.

    Args:
        config: Discord configuration.
    """
    bot = create_bot(config)
    await bot.start(config.bot_token)
