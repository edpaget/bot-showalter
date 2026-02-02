from unittest.mock import patch

from fantasy_baseball_manager.player_id.mapper import (
    _YAHOO_SPLIT_OVERRIDES,
    SfbbMapper,
    _parse_sfbb_csv,
    build_cached_sfbb_mapper,
    build_sfbb_mapper,
)

SAMPLE_CSV = (
    "IDPLAYER,PLAYERNAME,YAHOOID,IDFANGRAPHS,OTHERFIELD\n"
    "1,Mike Trout,10155,19054,xyz\n"
    "2,Shohei Ohtani,10835,19755,abc\n"
    "3,No Yahoo,,99999,def\n"
    "4,No FanGraphs,55555,,ghi\n"
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
