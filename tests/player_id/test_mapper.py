from unittest.mock import patch

from fantasy_baseball_manager.player.identity import Player
from fantasy_baseball_manager.player_id.mapper import (
    _YAHOO_SPLIT_OVERRIDES,
    PlayerMapperError,
    SfbbMapper,
    _parse_sfbb_csv,
    build_cached_sfbb_mapper,
    build_sfbb_mapper,
)

SAMPLE_CSV = (
    "IDPLAYER,PLAYERNAME,YAHOOID,IDFANGRAPHS,MLBID,OTHERFIELD\n"
    "1,Mike Trout,10155,19054,545361,xyz\n"
    "2,Shohei Ohtani,10835,19755,660271,abc\n"
    "3,No Yahoo,,99999,111111,def\n"
    "4,No FanGraphs,55555,,,ghi\n"
)


class TestSfbbMapper:
    def test_basic_lookup(self) -> None:
        mapper = SfbbMapper({"10155": "19054"}, {"19054": "10155"})
        assert mapper.yahoo_to_fangraphs("10155") == "19054"
        assert mapper.fangraphs_to_yahoo("19054") == "10155"

    def test_unknown_id_returns_none(self) -> None:
        mapper = SfbbMapper({"10155": "19054"}, {"19054": "10155"})
        assert mapper.yahoo_to_fangraphs("99999") is None
        assert mapper.fangraphs_to_yahoo("99999") is None

    def test_bidirectional(self) -> None:
        y2fg = {"y1": "fg1", "y2": "fg2"}
        fg2y = {"fg1": "y1", "fg2": "y2"}
        mapper = SfbbMapper(y2fg, fg2y)
        assert mapper.yahoo_to_fangraphs("y1") == "fg1"
        assert mapper.yahoo_to_fangraphs("y2") == "fg2"
        assert mapper.fangraphs_to_yahoo("fg1") == "y1"
        assert mapper.fangraphs_to_yahoo("fg2") == "y2"

    def test_empty_dicts(self) -> None:
        mapper = SfbbMapper({}, {})
        assert mapper.yahoo_to_fangraphs("y1") is None
        assert mapper.fangraphs_to_yahoo("fg1") is None

    def test_map_properties(self) -> None:
        y2fg = {"y1": "fg1", "y2": "fg2"}
        fg2y = {"fg1": "y1", "fg2": "y2"}
        mapper = SfbbMapper(y2fg, fg2y)
        assert mapper.yahoo_to_fg_map == {"y1": "fg1", "y2": "fg2"}
        assert mapper.fg_to_yahoo_map == {"fg1": "y1", "fg2": "y2"}

    def test_map_properties_return_copies(self) -> None:
        mapper = SfbbMapper({"y1": "fg1"}, {"fg1": "y1"})
        mapper.yahoo_to_fg_map["y2"] = "fg2"
        assert mapper.yahoo_to_fangraphs("y2") is None

    def test_fangraphs_to_mlbam_returns_mapped_id(self) -> None:
        mapper = SfbbMapper({}, {}, fg_to_mlbam={"19054": "545361"}, mlbam_to_fg={"545361": "19054"})
        assert mapper.fangraphs_to_mlbam("19054") == "545361"

    def test_mlbam_to_fangraphs_returns_mapped_id(self) -> None:
        mapper = SfbbMapper({}, {}, fg_to_mlbam={"19054": "545361"}, mlbam_to_fg={"545361": "19054"})
        assert mapper.mlbam_to_fangraphs("545361") == "19054"

    def test_unknown_mlbam_returns_none(self) -> None:
        mapper = SfbbMapper({}, {})
        assert mapper.fangraphs_to_mlbam("99999") is None
        assert mapper.mlbam_to_fangraphs("99999") is None


class TestParseSfbbCsv:
    def test_parses_valid_rows(self) -> None:
        mapper = _parse_sfbb_csv(SAMPLE_CSV)
        assert mapper.yahoo_to_fangraphs("10155") == "19054"
        assert mapper.yahoo_to_fangraphs("10835") == "19755"
        assert mapper.fangraphs_to_yahoo("19054") == "10155"
        assert mapper.fangraphs_to_yahoo("19755") == "10835"

    def test_skips_rows_missing_yahoo_id(self) -> None:
        mapper = _parse_sfbb_csv(SAMPLE_CSV)
        assert mapper.fangraphs_to_yahoo("99999") is None

    def test_skips_rows_missing_fangraphs_id(self) -> None:
        mapper = _parse_sfbb_csv(SAMPLE_CSV)
        assert mapper.yahoo_to_fangraphs("55555") is None

    def test_empty_csv(self) -> None:
        mapper = _parse_sfbb_csv("YAHOOID,IDFANGRAPHS\n")
        assert mapper.yahoo_to_fg_map == {}

    def test_parse_csv_builds_mlbam_mapping(self) -> None:
        mapper = _parse_sfbb_csv(SAMPLE_CSV)
        assert mapper.fangraphs_to_mlbam("19054") == "545361"
        assert mapper.fangraphs_to_mlbam("19755") == "660271"
        assert mapper.mlbam_to_fangraphs("545361") == "19054"
        assert mapper.mlbam_to_fangraphs("660271") == "19755"

    def test_parse_csv_skips_rows_missing_mlbam(self) -> None:
        mapper = _parse_sfbb_csv(SAMPLE_CSV)
        # Row 4 has no MLBID
        assert mapper.fangraphs_to_mlbam("") is None

    def test_parse_csv_mlbam_without_fangraphs_skipped(self) -> None:
        mapper = _parse_sfbb_csv(SAMPLE_CSV)
        # Row 3 has MLBID but no YAHOOID; MLBID should still map to FG ID
        assert mapper.mlbam_to_fangraphs("111111") == "99999"


class TestBuildSfbbMapper:
    @patch("fantasy_baseball_manager.player_id.mapper._download_sfbb_csv")
    def test_downloads_and_parses(self, mock_download: object) -> None:
        mock_download.return_value = SAMPLE_CSV  # type: ignore[union-attr]
        mapper = build_sfbb_mapper("http://fake-url")
        assert mapper.yahoo_to_fangraphs("10155") == "19054"
        mock_download.assert_called_once_with("http://fake-url")  # type: ignore[union-attr]


class FakeCacheStore:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def get(self, namespace: str, key: str) -> str | None:
        return self._data.get((namespace, key))

    def put(self, namespace: str, key: str, value: str, ttl_seconds: int) -> None:
        self._data[(namespace, key)] = value

    def invalidate(self, namespace: str, key: str | None = None) -> None:
        if key is not None:
            self._data.pop((namespace, key), None)
        else:
            to_remove = [k for k in self._data if k[0] == namespace]
            for k in to_remove:
                del self._data[k]


class TestBuildCachedSfbbMapper:
    @patch("fantasy_baseball_manager.player_id.mapper._download_sfbb_csv")
    def test_cache_miss_downloads_and_stores(self, mock_download: object) -> None:
        mock_download.return_value = SAMPLE_CSV  # type: ignore[union-attr]
        cache = FakeCacheStore()
        mapper = build_cached_sfbb_mapper(cache, "test_key", ttl=3600, csv_url="http://fake")

        assert mapper.yahoo_to_fangraphs("10155") == "19054"
        assert cache.get("sfbb_csv", "test_key") is not None
        mock_download.assert_called_once_with("http://fake")  # type: ignore[union-attr]

    @patch("fantasy_baseball_manager.player_id.mapper._download_sfbb_csv")
    def test_cache_hit_skips_download(self, mock_download: object) -> None:
        cache = FakeCacheStore()
        cache.put("sfbb_csv", "test_key", SAMPLE_CSV, 3600)

        mapper = build_cached_sfbb_mapper(cache, "test_key", ttl=3600, csv_url="http://fake")

        assert mapper.yahoo_to_fangraphs("10155") == "19054"
        mock_download.assert_not_called()  # type: ignore[union-attr]

    @patch("fantasy_baseball_manager.player_id.mapper._download_sfbb_csv")
    def test_cache_expiry_redownloads(self, mock_download: object) -> None:
        mock_download.return_value = SAMPLE_CSV  # type: ignore[union-attr]
        cache = FakeCacheStore()

        # First call populates cache
        build_cached_sfbb_mapper(cache, "test_key", ttl=3600, csv_url="http://fake")

        # Simulate expiry
        cache.invalidate("sfbb_csv", "test_key")

        # Second call should re-download
        mapper = build_cached_sfbb_mapper(cache, "test_key", ttl=3600, csv_url="http://fake")
        assert mapper.yahoo_to_fangraphs("10155") == "19054"
        assert mock_download.call_count == 2  # type: ignore[union-attr]


class TestSplitOverrides:
    def test_split_override_resolves_synthetic_to_real(self) -> None:
        """Synthetic Yahoo IDs (e.g. 1000001) resolve to same FG ID as real ID."""
        mapper = _parse_sfbb_csv(SAMPLE_CSV)
        # Real ID 10835 -> FG 19755
        assert mapper.yahoo_to_fangraphs("10835") == "19755"
        # Synthetic split IDs should also resolve
        for synthetic_id in _YAHOO_SPLIT_OVERRIDES:
            assert mapper.yahoo_to_fangraphs(synthetic_id) == "19755"

    def test_split_override_ignored_when_real_id_missing(self) -> None:
        """If the real Yahoo ID isn't in SFBB, synthetic IDs don't get mapped."""
        csv_without_ohtani = "IDPLAYER,PLAYERNAME,YAHOOID,IDFANGRAPHS\n" "1,Mike Trout,10155,19054\n"
        mapper = _parse_sfbb_csv(csv_without_ohtani)
        for synthetic_id in _YAHOO_SPLIT_OVERRIDES:
            assert mapper.yahoo_to_fangraphs(synthetic_id) is None


class TestSfbbMapperCallable:
    """Tests for the DataSource-style callable interface."""

    def test_call_single_player_enriches_ids(self) -> None:
        """Calling mapper with a single Player returns enriched Player."""
        mapper = SfbbMapper(
            {"10155": "19054"},
            {"19054": "10155"},
            fg_to_mlbam={"19054": "545361"},
            mlbam_to_fg={"545361": "19054"},
        )
        player = Player(name="Mike Trout", yahoo_id="10155")

        result = mapper(player)

        assert result.is_ok()
        enriched = result.unwrap()
        assert enriched.fangraphs_id == "19054"
        assert enriched.mlbam_id == "545361"
        assert enriched.name == "Mike Trout"
        assert enriched.yahoo_id == "10155"

    def test_call_single_player_unmapped_returns_unchanged(self) -> None:
        """Calling mapper with unmapped Player returns Player with None IDs."""
        mapper = SfbbMapper({}, {})
        player = Player(name="Unknown", yahoo_id="99999")

        result = mapper(player)

        assert result.is_ok()
        enriched = result.unwrap()
        assert enriched.fangraphs_id is None
        assert enriched.mlbam_id is None

    def test_call_list_enriches_all_players(self) -> None:
        """Calling mapper with list of Players returns list of enriched Players."""
        mapper = SfbbMapper(
            {"10155": "19054", "10835": "19755"},
            {"19054": "10155", "19755": "10835"},
            fg_to_mlbam={"19054": "545361", "19755": "660271"},
            mlbam_to_fg={"545361": "19054", "660271": "19755"},
        )
        players = [
            Player(name="Mike Trout", yahoo_id="10155"),
            Player(name="Shohei Ohtani", yahoo_id="10835"),
        ]

        result = mapper(players)

        assert result.is_ok()
        enriched_list = result.unwrap()
        assert len(enriched_list) == 2
        assert enriched_list[0].fangraphs_id == "19054"
        assert enriched_list[0].mlbam_id == "545361"
        assert enriched_list[1].fangraphs_id == "19755"
        assert enriched_list[1].mlbam_id == "660271"

    def test_call_empty_list_returns_empty_list(self) -> None:
        """Calling mapper with empty list returns empty list."""
        mapper = SfbbMapper({}, {})

        result = mapper([])

        assert result.is_ok()
        assert result.unwrap() == []

    def test_call_preserves_existing_player_fields(self) -> None:
        """Enrichment preserves all existing Player fields."""
        mapper = SfbbMapper({"10155": "19054"}, {"19054": "10155"})
        player = Player(
            name="Mike Trout",
            yahoo_id="10155",
            team="LAA",
            eligible_positions=("CF", "OF"),
            age=32,
        )

        result = mapper(player)

        assert result.is_ok()
        enriched = result.unwrap()
        assert enriched.team == "LAA"
        assert enriched.eligible_positions == ("CF", "OF")
        assert enriched.age == 32

    def test_call_does_not_override_existing_ids(self) -> None:
        """If Player already has IDs, they are preserved (not overridden by None)."""
        mapper = SfbbMapper({}, {})  # No mappings
        player = Player(
            name="Mike Trout",
            yahoo_id="10155",
            fangraphs_id="already_set",
            mlbam_id="also_set",
        )

        result = mapper(player)

        assert result.is_ok()
        enriched = result.unwrap()
        # with_ids only overrides with non-None, so existing values preserved
        assert enriched.fangraphs_id == "already_set"
        assert enriched.mlbam_id == "also_set"

    def test_call_two_way_player_uses_synthetic_id(self) -> None:
        """Two-way players use yahoo_sub_id (synthetic) for lookup."""
        # Synthetic ID 1000001 maps to Ohtani's real ID in the sample CSV
        mapper = _parse_sfbb_csv(SAMPLE_CSV)
        player = Player(
            name="Shohei Ohtani",
            yahoo_id="10835",
            yahoo_sub_id="1000001",  # Synthetic batter ID
        )

        result = mapper(player)

        assert result.is_ok()
        enriched = result.unwrap()
        # Synthetic ID 1000001 -> FG 19755 (via split override)
        assert enriched.fangraphs_id == "19755"
        assert enriched.mlbam_id == "660271"

    def test_call_mixed_mapped_unmapped(self) -> None:
        """List with both mapped and unmapped players handles both correctly."""
        mapper = SfbbMapper({"10155": "19054"}, {"19054": "10155"})
        players = [
            Player(name="Mike Trout", yahoo_id="10155"),
            Player(name="Unknown", yahoo_id="99999"),
        ]

        result = mapper(players)

        assert result.is_ok()
        enriched_list = result.unwrap()
        assert enriched_list[0].fangraphs_id == "19054"
        assert enriched_list[1].fangraphs_id is None


class TestPlayerMapperError:
    """Tests for the PlayerMapperError exception."""

    def test_error_with_message_only(self) -> None:
        error = PlayerMapperError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.cause is None

    def test_error_with_cause(self) -> None:
        cause = ValueError("Original error")
        error = PlayerMapperError("Wrapper error", cause=cause)
        assert str(error) == "Wrapper error: Original error"
        assert error.message == "Wrapper error"
        assert error.cause is cause
