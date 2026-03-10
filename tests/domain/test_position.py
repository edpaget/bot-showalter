import pytest

from fantasy_baseball_manager.domain.position import (
    OF_POSITIONS,
    Position,
    consolidate_outfield,
    position_from_raw,
)


class TestFromRaw:
    def test_uppercase(self) -> None:
        assert position_from_raw("C") == Position.C

    def test_lowercase(self) -> None:
        assert position_from_raw("c") == Position.C

    def test_mixed_case(self) -> None:
        assert position_from_raw("Ss") == Position.SS

    def test_multi_char(self) -> None:
        assert position_from_raw("1b") == Position.FIRST_BASE

    def test_util(self) -> None:
        assert position_from_raw("util") == Position.UTIL

    def test_bench(self) -> None:
        assert position_from_raw("BN") == Position.BN

    def test_whitespace_stripped(self) -> None:
        assert position_from_raw(" SP ") == Position.SP

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown position"):
            position_from_raw("XX")


class TestConsolidateOutfield:
    def test_lf_to_of(self) -> None:
        assert consolidate_outfield(Position.LF) == Position.OF

    def test_cf_to_of(self) -> None:
        assert consolidate_outfield(Position.CF) == Position.OF

    def test_rf_to_of(self) -> None:
        assert consolidate_outfield(Position.RF) == Position.OF

    def test_of_stays_of(self) -> None:
        assert consolidate_outfield(Position.OF) == Position.OF

    def test_non_of_unchanged(self) -> None:
        assert consolidate_outfield(Position.C) == Position.C
        assert consolidate_outfield(Position.SP) == Position.SP


class TestOfPositions:
    def test_contains_expected(self) -> None:
        assert frozenset({Position.LF, Position.CF, Position.RF, Position.OF}) == OF_POSITIONS
