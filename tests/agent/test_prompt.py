from fantasy_baseball_manager.agent.prompt import SYSTEM_PROMPT


class TestSystemPrompt:
    def test_is_non_empty_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_mentions_fantasy_baseball(self) -> None:
        assert "fantasy baseball" in SYSTEM_PROMPT.lower()

    def test_mentions_key_tool_names(self) -> None:
        expected_tools = [
            "search_players",
            "get_player_bio",
            "lookup_projections",
            "lookup_valuations",
            "get_rankings",
            "get_value_over_adp",
            "get_overperformers",
            "get_underperformers",
            "find_players",
        ]
        for tool_name in expected_tools:
            assert tool_name in SYSTEM_PROMPT, f"Missing tool: {tool_name}"

    def test_instructs_citing_numbers(self) -> None:
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "cite" in prompt_lower or "specific numbers" in prompt_lower
