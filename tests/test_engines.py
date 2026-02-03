import pytest
import typer

from fantasy_baseball_manager.engines import (
    DEFAULT_ENGINE,
    DEFAULT_METHOD,
    SUPPORTED_ENGINES,
    SUPPORTED_METHODS,
    validate_engine,
    validate_method,
)


class TestValidateEngine:
    def test_marcel_accepted(self) -> None:
        validate_engine("marcel")

    def test_unknown_engine_exits(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            validate_engine("steamer")
        assert exc_info.value.exit_code == 1

    def test_default_engine_is_marcel(self) -> None:
        assert DEFAULT_ENGINE == "marcel"

    def test_supported_engines_contains_marcel(self) -> None:
        assert "marcel" in SUPPORTED_ENGINES

    def test_marcel_classic_accepted(self) -> None:
        validate_engine("marcel_classic")

    def test_marcel_full_accepted(self) -> None:
        validate_engine("marcel_full")

    def test_supported_engines_contains_all_variants(self) -> None:
        for name in ("marcel_classic", "marcel", "marcel_full"):
            assert name in SUPPORTED_ENGINES

    def test_supported_engines_has_exactly_three(self) -> None:
        assert len(SUPPORTED_ENGINES) == 3


class TestValidateMethod:
    def test_zscore_accepted(self) -> None:
        validate_method("zscore")

    def test_unknown_method_exits(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            validate_method("sgp")
        assert exc_info.value.exit_code == 1

    def test_default_method_is_zscore(self) -> None:
        assert DEFAULT_METHOD == "zscore"

    def test_supported_methods_contains_zscore(self) -> None:
        assert "zscore" in SUPPORTED_METHODS
