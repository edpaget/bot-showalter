from fantasy_baseball_manager.yahoo.player_parsing import extract_player_data


class TestExtractPlayerData:
    def test_extracts_all_fields(self) -> None:
        player_meta = [
            {"player_key": "449.p.12345"},
            {"name": {"full": "Mike Trout"}},
            {"editorial_team_abbr": "LAA"},
            {"eligible_positions": [{"position": "CF"}, {"position": "Util"}]},
            {"player_id": "12345"},
        ]

        result = extract_player_data(player_meta)

        assert result["player_key"] == "449.p.12345"
        assert result["name"] == "Mike Trout"
        assert result["editorial_team_abbr"] == "LAA"
        assert result["eligible_positions"] == ["CF", "UTIL"]
        assert result["player_id"] == "12345"

    def test_skips_non_dict_items(self) -> None:
        player_meta = [
            "some_string",
            42,
            {"player_key": "449.p.99"},
        ]

        result = extract_player_data(player_meta)

        assert result == {"player_key": "449.p.99"}

    def test_empty_list(self) -> None:
        assert extract_player_data([]) == {}

    def test_filters_non_dict_positions(self) -> None:
        player_meta = [
            {"eligible_positions": [{"position": "SS"}, "count", {"position": "Util"}]},
        ]

        result = extract_player_data(player_meta)

        assert result["eligible_positions"] == ["SS", "UTIL"]

    def test_filters_non_position_entries(self) -> None:
        player_meta = [
            {
                "eligible_positions": [
                    {"position": "CF"},
                    {"position": "OF"},
                    {"position": "Util"},
                    {"position": "BN"},
                    {"position": "IL"},
                    {"position": "IL+"},
                ]
            },
        ]

        result = extract_player_data(player_meta)

        assert result["eligible_positions"] == ["CF", "OF", "UTIL"]
