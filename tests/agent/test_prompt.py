import datetime

from fantasy_baseball_manager.agent.prompt import build_system_prompt, current_season


class TestCurrentSeason:
    def test_mid_season_returns_current_year(self) -> None:
        assert current_season(datetime.date(2025, 6, 15)) == 2025

    def test_january_returns_current_year(self) -> None:
        assert current_season(datetime.date(2026, 1, 1)) == 2026

    def test_september_returns_current_year(self) -> None:
        assert current_season(datetime.date(2025, 9, 30)) == 2025

    def test_october_returns_next_year(self) -> None:
        assert current_season(datetime.date(2025, 10, 1)) == 2026

    def test_december_returns_next_year(self) -> None:
        assert current_season(datetime.date(2025, 12, 31)) == 2026

    def test_defaults_to_today(self) -> None:
        today = datetime.date.today()
        expected = today.year + 1 if today.month >= 10 else today.year
        assert current_season() == expected


class TestBuildSystemPrompt:
    def test_is_non_empty_string(self) -> None:
        prompt = build_system_prompt(2025)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_season(self) -> None:
        prompt = build_system_prompt(2025)
        assert "2025" in prompt

    def test_different_season(self) -> None:
        prompt = build_system_prompt(2023)
        assert "2023" in prompt
        assert "2025" not in prompt

    def test_mentions_fantasy_baseball(self) -> None:
        prompt = build_system_prompt(2025)
        assert "fantasy baseball" in prompt.lower()

    def test_mentions_key_tool_names(self) -> None:
        prompt = build_system_prompt(2025)
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
            assert tool_name in prompt, f"Missing tool: {tool_name}"

    def test_instructs_citing_numbers(self) -> None:
        prompt = build_system_prompt(2025).lower()
        assert "cite" in prompt or "specific numbers" in prompt

    def test_mentions_multiple_seasons(self) -> None:
        """The prompt should indicate the agent can look up data across seasons."""
        prompt = build_system_prompt(2025).lower()
        assert "any" in prompt or "multiple seasons" in prompt
