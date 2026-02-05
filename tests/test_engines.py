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
    def test_supported_engines_is_non_empty(self) -> None:
        assert len(SUPPORTED_ENGINES) > 0

    def test_default_engine_is_supported(self) -> None:
        assert DEFAULT_ENGINE in SUPPORTED_ENGINES

    def test_all_supported_engines_pass_validation(self) -> None:
        for engine in SUPPORTED_ENGINES:
            validate_engine(engine)  # Should not raise

    def test_unknown_engine_exits(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            validate_engine("not_a_real_engine")
        assert exc_info.value.exit_code == 1


class TestValidateMethod:
    def test_supported_methods_is_non_empty(self) -> None:
        assert len(SUPPORTED_METHODS) > 0

    def test_default_method_is_supported(self) -> None:
        assert DEFAULT_METHOD in SUPPORTED_METHODS

    def test_all_supported_methods_pass_validation(self) -> None:
        for method in SUPPORTED_METHODS:
            validate_method(method)  # Should not raise

    def test_unknown_method_exits(self) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            validate_method("not_a_real_method")
        assert exc_info.value.exit_code == 1
