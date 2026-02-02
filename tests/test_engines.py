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

    def test_marcel_park_accepted(self) -> None:
        validate_engine("marcel_park")

    def test_marcel_statreg_accepted(self) -> None:
        validate_engine("marcel_statreg")

    def test_marcel_plus_accepted(self) -> None:
        validate_engine("marcel_plus")

    def test_supported_engines_contains_all_variants(self) -> None:
        for name in ("marcel", "marcel_park", "marcel_statreg", "marcel_plus"):
            assert name in SUPPORTED_ENGINES


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
