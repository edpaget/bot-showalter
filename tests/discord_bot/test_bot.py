from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessageChunk

from fantasy_baseball_manager.discord_bot.bot import FBMDiscordBot


def _make_bot() -> FBMDiscordBot:
    agent = MagicMock()
    return FBMDiscordBot(agent)


def _make_message(
    *,
    content: str = "hello",
    author_bot: bool = False,
    mentions_bot: bool = True,
    bot_id: int = 12345,
) -> tuple[MagicMock, MagicMock]:
    """Build a mock discord.Message and bot user."""
    msg = MagicMock()
    msg.content = content
    msg.author.bot = author_bot
    msg.channel.send = AsyncMock()
    msg.channel.typing = MagicMock(return_value=AsyncMock())
    # Make typing() usable as async context manager
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=None)
    ctx.__aexit__ = AsyncMock(return_value=None)
    msg.channel.typing.return_value = ctx

    bot_user = MagicMock()
    bot_user.id = bot_id
    if mentions_bot:
        msg.mentions = [bot_user]
    else:
        msg.mentions = []
    return msg, bot_user


class TestFBMDiscordBot:
    def test_ignores_bot_messages(self) -> None:
        bot = _make_bot()
        msg, bot_user = _make_message(author_bot=True, mentions_bot=True)
        bot._connection.user = bot_user

        asyncio.run(bot.on_message(msg))
        msg.channel.send.assert_not_called()

    def test_ignores_non_mention_messages(self) -> None:
        bot = _make_bot()
        msg, bot_user = _make_message(mentions_bot=False)
        bot._connection.user = bot_user

        asyncio.run(bot.on_message(msg))
        msg.channel.send.assert_not_called()

    def test_ignores_own_messages(self) -> None:
        bot = _make_bot()
        msg, bot_user = _make_message(mentions_bot=True)
        msg.author = bot_user
        bot._connection.user = bot_user

        asyncio.run(bot.on_message(msg))
        msg.channel.send.assert_not_called()

    def test_responds_to_mention(self) -> None:
        agent = MagicMock()
        chunk = MagicMock(spec=AIMessageChunk)
        chunk.content = "I can help!"
        agent.stream.return_value = [(chunk, {"langgraph_node": "agent"})]

        bot = FBMDiscordBot(agent)
        bot_id = 12345
        msg, bot_user = _make_message(content=f"<@{bot_id}> what's up", bot_id=bot_id)
        bot._connection.user = bot_user

        asyncio.run(bot.on_message(msg))
        msg.channel.send.assert_called_once_with("I can help!")

    def test_strips_mention_from_text(self) -> None:
        agent = MagicMock()
        chunk = MagicMock(spec=AIMessageChunk)
        chunk.content = "response"
        agent.stream.return_value = [(chunk, {"langgraph_node": "agent"})]

        bot = FBMDiscordBot(agent)
        bot_id = 99999
        msg, bot_user = _make_message(content=f"<@{bot_id}> tell me about Mike Trout", bot_id=bot_id)
        bot._connection.user = bot_user

        asyncio.run(bot.on_message(msg))
        # Verify the agent was called with mention stripped
        call_args = agent.stream.call_args
        messages = call_args[0][0]["messages"]
        assert messages == [("user", "tell me about Mike Trout")]

    def test_sends_typing_indicator(self) -> None:
        agent = MagicMock()
        chunk = MagicMock(spec=AIMessageChunk)
        chunk.content = "answer"
        agent.stream.return_value = [(chunk, {"langgraph_node": "agent"})]

        bot = FBMDiscordBot(agent)
        msg, bot_user = _make_message(content="<@123> hi", bot_id=123)
        bot._connection.user = bot_user

        asyncio.run(bot.on_message(msg))
        msg.channel.typing.assert_called_once()

    def test_splits_long_response(self) -> None:
        agent = MagicMock()
        first_para = "a" * 1500
        second_para = "b" * 1500
        chunk = MagicMock(spec=AIMessageChunk)
        chunk.content = first_para + "\n\n" + second_para
        agent.stream.return_value = [(chunk, {"langgraph_node": "agent"})]

        bot = FBMDiscordBot(agent)
        msg, bot_user = _make_message(content="<@123> hello", bot_id=123)
        bot._connection.user = bot_user

        asyncio.run(bot.on_message(msg))
        assert msg.channel.send.call_count == 2
        msg.channel.send.assert_any_call(first_para)
        msg.channel.send.assert_any_call(second_para)
