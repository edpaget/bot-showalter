"""Tests for name normalization utilities."""

from fantasy_baseball_manager.adp.name_utils import normalize_name


class TestNormalizeName:
    """Tests for normalize_name function."""

    def test_accents_removed(self) -> None:
        assert normalize_name("José Ramírez") == "jose ramirez"

    def test_suffixes_stripped(self) -> None:
        assert normalize_name("Ohtani (Batter)") == "ohtani"
        assert normalize_name("Ohtani (Pitcher)") == "ohtani"

    def test_periods_removed(self) -> None:
        assert normalize_name("J.D. Martinez") == "jd martinez"

    def test_combined_jr(self) -> None:
        assert normalize_name("Vladimir Guerrero Jr.") == "vladimir guerrero jr"

    def test_lowercase(self) -> None:
        assert normalize_name("Mike Trout") == "mike trout"

    def test_already_normalized(self) -> None:
        assert normalize_name("mike trout") == "mike trout"

    def test_suffix_with_spaces(self) -> None:
        assert normalize_name("Shohei Ohtani  (Batter) ") == "shohei ohtani"

    def test_no_suffix_match_leaves_name(self) -> None:
        assert normalize_name("Player (Notes)") == "player (notes)"
