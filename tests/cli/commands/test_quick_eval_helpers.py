"""Tests for helper functions in cli/commands/quick_eval.py."""

from fantasy_baseball_manager.cli.commands.quick_eval import _coerce_value, _parse_params


class TestCoerceValue:
    def test_bool_true(self) -> None:
        assert _coerce_value("true") is True

    def test_bool_true_mixed_case(self) -> None:
        assert _coerce_value("True") is True

    def test_bool_false(self) -> None:
        assert _coerce_value("false") is False

    def test_int(self) -> None:
        assert _coerce_value("42") == 42
        assert isinstance(_coerce_value("42"), int)

    def test_float(self) -> None:
        assert _coerce_value("3.14") == 3.14
        assert isinstance(_coerce_value("3.14"), float)

    def test_string_fallback(self) -> None:
        assert _coerce_value("hello") == "hello"
        assert isinstance(_coerce_value("hello"), str)


class TestParseParams:
    def test_none_returns_none(self) -> None:
        assert _parse_params(None) is None

    def test_empty_list_returns_none(self) -> None:
        assert _parse_params([]) is None

    def test_key_value_pairs(self) -> None:
        result = _parse_params(["n_estimators=100", "learning_rate=0.1", "verbose=true"])
        assert result == {"n_estimators": 100, "learning_rate": 0.1, "verbose": True}

    def test_string_value(self) -> None:
        result = _parse_params(["objective=regression"])
        assert result == {"objective": "regression"}
