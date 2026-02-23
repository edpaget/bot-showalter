from __future__ import annotations

import logging
import re

import discord
from langgraph.graph.state import CompiledStateGraph

from fantasy_baseball_manager.discord_bot.agent_handler import handle_message

logger = logging.getLogger(__name__)


class FBMDiscordBot(discord.Client):
    """Discord client that responds to @-mentions with agent-powered replies."""

    def __init__(self, agent: CompiledStateGraph) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.agent = agent

    async def on_ready(self) -> None:  # pragma: no cover
        logger.info("Discord bot logged in as %s", self.user)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if self.user is None:
            return  # pragma: no cover
        if message.author == self.user:
            return
        if self.user not in message.mentions:
            return

        text = re.sub(rf"<@!?{self.user.id}>", "", message.content).strip()
        async with message.channel.typing():
            chunks = await handle_message(self.agent, text)
        for chunk in chunks:
            await message.channel.send(chunk)
