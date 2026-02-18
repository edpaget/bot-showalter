from fantasy_baseball_manager.domain.errors import (
    ConfigError,
    DispatchError,
    FbmError,
    IngestError,
)


class TestFbmError:
    def test_construction(self) -> None:
        err = FbmError(message="something went wrong")
        assert err.message == "something went wrong"

    def test_frozen(self) -> None:
        err = FbmError(message="x")
        try:
            err.message = "y"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass


class TestDispatchError:
    def test_construction(self) -> None:
        err = DispatchError(message="unsupported", model_name="marcel", operation="train")
        assert err.message == "unsupported"
        assert err.model_name == "marcel"
        assert err.operation == "train"

    def test_inherits_fbm_error(self) -> None:
        err = DispatchError(message="x", model_name="m", operation="o")
        assert isinstance(err, FbmError)


class TestIngestError:
    def test_construction(self) -> None:
        err = IngestError(
            message="fetch failed",
            source_type="pybaseball",
            source_detail="fg_batting_data",
            target_table="batting_stats",
        )
        assert err.message == "fetch failed"
        assert err.source_type == "pybaseball"
        assert err.source_detail == "fg_batting_data"
        assert err.target_table == "batting_stats"

    def test_inherits_fbm_error(self) -> None:
        err = IngestError(message="x", source_type="t", source_detail="d", target_table="tbl")
        assert isinstance(err, FbmError)


class TestConfigError:
    def test_construction_defaults(self) -> None:
        err = ConfigError(message="bad config")
        assert err.message == "bad config"
        assert err.unrecognized_keys == ()

    def test_construction_with_keys(self) -> None:
        err = ConfigError(message="bad", unrecognized_keys=("foo", "bar"))
        assert err.unrecognized_keys == ("foo", "bar")

    def test_inherits_fbm_error(self) -> None:
        err = ConfigError(message="x")
        assert isinstance(err, FbmError)
