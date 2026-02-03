"""Tests for the Discord bot."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import discord
from fantasy_baseball_manager.discord.bot import FantasyBaseballBot, create_bot
from fantasy_baseball_manager.discord.config import (
    ConfigurationError,
    DiscordConfig,
    load_discord_config,
)


class TestDiscordConfig:
    """Tests for DiscordConfig and load_discord_config."""

    def test_load_discord_config_with_token(self) -> None:
        """load_discord_config returns config when token is set."""
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}, clear=False):
            # Clear DISCORD_ALLOWED_CHANNELS if it exists
            env = os.environ.copy()
            env.pop("DISCORD_ALLOWED_CHANNELS", None)
            with patch.dict(os.environ, env, clear=True):
                os.environ["DISCORD_BOT_TOKEN"] = "test-token"
                config = load_discord_config()

        assert config.bot_token == "test-token"
        assert config.allowed_channels is None

    def test_load_discord_config_with_channels(self) -> None:
        """load_discord_config parses allowed channels."""
        with patch.dict(
            os.environ,
            {
                "DISCORD_BOT_TOKEN": "test-token",
                "DISCORD_ALLOWED_CHANNELS": "123,456,789",
            },
            clear=True,
        ):
            config = load_discord_config()

        assert config.bot_token == "test-token"
        assert config.allowed_channels == (123, 456, 789)

    def test_load_discord_config_missing_token_raises(self) -> None:
        """load_discord_config raises ConfigurationError when token is missing."""
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ConfigurationError, match="DISCORD_BOT_TOKEN"):
            load_discord_config()

    def test_load_discord_config_invalid_channels_raises(self) -> None:
        """load_discord_config raises ConfigurationError for invalid channel IDs."""
        with (
            patch.dict(
                os.environ,
                {
                    "DISCORD_BOT_TOKEN": "test-token",
                    "DISCORD_ALLOWED_CHANNELS": "123,not-a-number,789",
                },
                clear=True,
            ),
            pytest.raises(ConfigurationError, match="comma-separated integers"),
        ):
            load_discord_config()


class TestFantasyBaseballBot:
    """Tests for FantasyBaseballBot."""

    def test_create_bot_returns_instance(self, discord_config: DiscordConfig) -> None:
        """create_bot returns a FantasyBaseballBot instance."""
        bot = create_bot(discord_config)
        assert isinstance(bot, FantasyBaseballBot)

    def test_bot_has_message_content_intent(self, discord_config: DiscordConfig) -> None:
        """Bot has message_content intent enabled."""
        bot = create_bot(discord_config)
        assert bot.intents.message_content is True

    def test_extract_message_content_removes_mention(self, discord_config: DiscordConfig) -> None:
        """_extract_message_content removes bot mention from message."""
        bot = create_bot(discord_config)

        # Mock the bot's user
        bot_user = MagicMock()
        bot_user.id = 123456
        bot._connection.user = bot_user

        message = MagicMock()
        message.content = "<@123456> What are the top batters?"

        result = bot._extract_message_content(message)
        assert result == "What are the top batters?"

    def test_extract_message_content_removes_nickname_mention(self, discord_config: DiscordConfig) -> None:
        """_extract_message_content removes nickname-style mention."""
        bot = create_bot(discord_config)

        bot_user = MagicMock()
        bot_user.id = 123456
        bot._connection.user = bot_user

        message = MagicMock()
        message.content = "<@!123456> Tell me about Shohei Ohtani"

        result = bot._extract_message_content(message)
        assert result == "Tell me about Shohei Ohtani"

    def test_truncate_response_short_message(self, discord_config: DiscordConfig) -> None:
        """_truncate_response returns short messages unchanged."""
        bot = create_bot(discord_config)
        content = "This is a short message"
        result = bot._truncate_response(content)
        assert result == content

    def test_truncate_response_long_message(self, discord_config: DiscordConfig) -> None:
        """_truncate_response truncates messages over 2000 characters."""
        bot = create_bot(discord_config)
        content = "x" * 2500
        result = bot._truncate_response(content)
        assert len(result) == 2000
        assert result.endswith("...")

    async def test_on_message_ignores_own_messages(self, discord_config: DiscordConfig) -> None:
        """Bot ignores its own messages."""
        bot = create_bot(discord_config)

        bot_user = MagicMock()
        bot._connection.user = bot_user

        message = MagicMock()
        message.author = bot_user
        message.reply = AsyncMock()

        await bot.on_message(message)

        message.reply.assert_not_called()

    async def test_on_message_ignores_non_mentions(self, discord_config: DiscordConfig) -> None:
        """Bot ignores messages that don't mention it."""
        bot = create_bot(discord_config)

        bot_user = MagicMock()
        bot._connection.user = bot_user

        message = MagicMock()
        message.author = MagicMock()
        message.mentions = []  # Bot not mentioned
        message.reply = AsyncMock()

        await bot.on_message(message)

        message.reply.assert_not_called()

    async def test_on_message_requires_thread(self, discord_config: DiscordConfig) -> None:
        """Bot prompts user to use a thread when mentioned outside of one."""
        bot = create_bot(discord_config)

        bot_user = MagicMock()
        bot._connection.user = bot_user

        message = MagicMock()
        message.author = MagicMock()
        message.mentions = [bot_user]
        # Channel is not a Thread
        message.channel = MagicMock(spec=discord.TextChannel)
        message.reply = AsyncMock()

        await bot.on_message(message)

        message.reply.assert_called_once()
        call_args = message.reply.call_args
        assert "thread" in call_args[0][0].lower()

    async def test_on_message_checks_allowed_channels(self, discord_config: DiscordConfig) -> None:
        """Bot ignores messages in non-allowed channels."""
        bot = create_bot(discord_config)

        bot_user = MagicMock()
        bot._connection.user = bot_user

        message = MagicMock()
        message.author = MagicMock()
        message.mentions = [bot_user]
        message.content = "<@123> Hello"

        # Thread in non-allowed channel
        thread = MagicMock(spec=discord.Thread)
        thread.id = 999
        thread.parent_id = 555555555  # Not in allowed_channels
        message.channel = thread
        message.reply = AsyncMock()

        await bot.on_message(message)

        # Should not reply since channel is not allowed
        message.reply.assert_not_called()

    async def test_on_message_requires_content(self, discord_config_no_channels: DiscordConfig) -> None:
        """Bot asks for content when mention has no message."""
        bot = create_bot(discord_config_no_channels)

        bot_user = MagicMock()
        bot_user.id = 123
        bot._connection.user = bot_user

        message = MagicMock()
        message.author = MagicMock()
        message.mentions = [bot_user]
        message.content = "<@123>"  # Just a mention, no content

        thread = MagicMock(spec=discord.Thread)
        thread.id = 999
        thread.parent_id = 123456789
        message.channel = thread
        message.reply = AsyncMock()

        await bot.on_message(message)

        message.reply.assert_called_once()
        call_args = message.reply.call_args
        assert "message" in call_args[0][0].lower()

    async def test_get_or_create_agent_caches_by_thread(
        self, discord_config: DiscordConfig
    ) -> None:
        """_get_or_create_agent returns the same agent for the same thread."""
        bot = create_bot(discord_config)

        with patch("fantasy_baseball_manager.discord.bot.create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            agent1 = await bot._get_or_create_agent("thread-123")
            agent2 = await bot._get_or_create_agent("thread-123")

            assert agent1 is agent2
            mock_create.assert_called_once_with(thread_id="thread-123")

    async def test_get_or_create_agent_different_threads(
        self, discord_config: DiscordConfig
    ) -> None:
        """_get_or_create_agent creates different agents for different threads."""
        bot = create_bot(discord_config)

        with patch("fantasy_baseball_manager.discord.bot.create_agent") as mock_create:
            mock_create.side_effect = [MagicMock(), MagicMock()]

            agent1 = await bot._get_or_create_agent("thread-123")
            agent2 = await bot._get_or_create_agent("thread-456")

            assert agent1 is not agent2
            assert mock_create.call_count == 2
