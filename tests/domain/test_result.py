from fantasy_baseball_manager.domain.result import Err, Ok, Result


class TestOk:
    def test_construction(self) -> None:
        ok: Ok[int] = Ok(42)
        assert ok.value == 42

    def test_equality(self) -> None:
        assert Ok(1) == Ok(1)
        assert Ok(1) != Ok(2)

    def test_frozen(self) -> None:
        ok = Ok(1)
        try:
            ok.value = 2  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass


class TestErr:
    def test_construction(self) -> None:
        err: Err[str] = Err("bad")
        assert err.error == "bad"

    def test_equality(self) -> None:
        assert Err("a") == Err("a")
        assert Err("a") != Err("b")

    def test_frozen(self) -> None:
        err = Err("x")
        try:
            err.error = "y"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass


class TestPatternMatching:
    def test_match_ok(self) -> None:
        result: Result[int, str] = Ok(10)
        match result:
            case Ok(value):
                assert value == 10
            case Err():
                raise AssertionError("Should not match Err")

    def test_match_err(self) -> None:
        result: Result[int, str] = Err("fail")
        match result:
            case Ok():
                raise AssertionError("Should not match Ok")
            case Err(error):
                assert error == "fail"

    def test_isinstance_narrowing(self) -> None:
        result: Result[int, str] = Ok(5)
        assert isinstance(result, Ok)
        assert not isinstance(result, Err)

        result2: Result[int, str] = Err("x")
        assert isinstance(result2, Err)
        assert not isinstance(result2, Ok)
