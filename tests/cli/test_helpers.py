import pytest
from click.exceptions import Exit

from fantasy_baseball_manager.cli._helpers import (
    coerce_value,
    parse_params,
    parse_system_version,
    parse_tags,
    set_nested,
)


class TestParseSystemVersion:
    def test_valid_system_version(self) -> None:
        assert parse_system_version("steamer/2025.1") == ("steamer", "2025.1")

    def test_missing_slash_exits(self) -> None:
        with pytest.raises(Exit):
            parse_system_version("steamer")

    def test_multiple_slashes_keeps_remainder(self) -> None:
        assert parse_system_version("a/b/c") == ("a", "b/c")


class TestCoerceValue:
    def test_bool_true(self) -> None:
        assert coerce_value("True") is True

    def test_bool_false(self) -> None:
        assert coerce_value("false") is False

    def test_int(self) -> None:
        assert coerce_value("42") == 42

    def test_float(self) -> None:
        assert coerce_value("0.1") == 0.1

    def test_string(self) -> None:
        assert coerce_value("hello") == "hello"


class TestSetNested:
    def test_flat_key(self) -> None:
        target: dict[str, object] = {}
        set_nested(target, "alpha", 0.1)
        assert target == {"alpha": 0.1}

    def test_single_dot(self) -> None:
        target: dict[str, object] = {}
        set_nested(target, "pitcher.learning_rate", 0.05)
        assert target == {"pitcher": {"learning_rate": 0.05}}

    def test_multiple_dots(self) -> None:
        target: dict[str, object] = {}
        set_nested(target, "a.b.c", 42)
        assert target == {"a": {"b": {"c": 42}}}

    def test_preserves_siblings(self) -> None:
        target: dict[str, object] = {"pitcher": {"n_estimators": 100}}
        set_nested(target, "pitcher.learning_rate", 0.05)
        assert target == {"pitcher": {"n_estimators": 100, "learning_rate": 0.05}}

    def test_overwrites_existing(self) -> None:
        target: dict[str, object] = {"pitcher": {"learning_rate": 0.1}}
        set_nested(target, "pitcher.learning_rate", 0.05)
        assert target == {"pitcher": {"learning_rate": 0.05}}


class TestParseParams:
    def test_parse_params_coerces_bool(self) -> None:
        result = parse_params(["use_playing_time=false"])
        assert result == {"use_playing_time": False}

    def test_parse_params_coerces_bool_true(self) -> None:
        result = parse_params(["use_playing_time=True"])
        assert result == {"use_playing_time": True}

    def test_parse_params_coerces_int(self) -> None:
        result = parse_params(["lags=5"])
        assert result == {"lags": 5}

    def test_parse_params_coerces_float(self) -> None:
        result = parse_params(["alpha=0.1"])
        assert result == {"alpha": 0.1}

    def test_parse_params_leaves_string(self) -> None:
        result = parse_params(["name=hello"])
        assert result == {"name": "hello"}

    def test_parse_params_none_returns_none(self) -> None:
        assert parse_params(None) is None

    def test_parse_params_dotted_key(self) -> None:
        result = parse_params(["pitcher.learning_rate=0.05"])
        assert result == {"pitcher": {"learning_rate": 0.05}}

    def test_parse_params_multiple_same_prefix(self) -> None:
        result = parse_params(["pitcher.learning_rate=0.05", "pitcher.n_estimators=200"])
        assert result == {"pitcher": {"learning_rate": 0.05, "n_estimators": 200}}

    def test_parse_params_mixed_flat_and_dotted(self) -> None:
        result = parse_params(["mode=preseason", "pitcher.learning_rate=0.05"])
        assert result == {"mode": "preseason", "pitcher": {"learning_rate": 0.05}}


class TestParseTags:
    def test_parse_tags_basic(self) -> None:
        assert parse_tags(["env=test"]) == {"env": "test"}

    def test_parse_tags_none(self) -> None:
        assert parse_tags(None) is None

    def test_parse_tags_multiple(self) -> None:
        assert parse_tags(["env=test", "run=1"]) == {"env": "test", "run": "1"}
