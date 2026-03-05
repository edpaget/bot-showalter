from fantasy_baseball_manager.name_utils import normalize_name, strip_name_decorations


class TestNormalizeName:
    def test_plain_name(self) -> None:
        assert normalize_name("Mike Trout") == "mike trout"

    def test_strips_jr_suffix(self) -> None:
        assert normalize_name("Bobby Witt Jr.") == "bobby witt"

    def test_strips_roman_numeral_suffix(self) -> None:
        assert normalize_name("Ken Griffey II") == "ken griffey"

    def test_strips_batter_parenthetical(self) -> None:
        assert normalize_name("Shohei Ohtani (Batter)") == "shohei ohtani"

    def test_strips_pitcher_parenthetical(self) -> None:
        assert normalize_name("Shohei Ohtani (Pitcher)") == "shohei ohtani"

    def test_strips_accents(self) -> None:
        assert normalize_name("Ronald Acuña") == "ronald acuna"

    def test_merges_dotted_initials(self) -> None:
        assert normalize_name("J.T. Realmuto") == "jt realmuto"

    def test_merges_spaced_dotted_initials(self) -> None:
        assert normalize_name("J. T. Realmuto") == "jt realmuto"

    def test_applies_nickname_alias(self) -> None:
        assert normalize_name("Matthew Boyd") == "matt boyd"
        assert normalize_name("Michael King") == "mike king"

    def test_combined_suffix_and_accent(self) -> None:
        assert normalize_name("Ronald Acuña Jr.") == "ronald acuna"


class TestStripNameDecorations:
    def test_strips_jr_suffix(self) -> None:
        assert strip_name_decorations("Bobby Witt Jr.") == "Bobby Witt"

    def test_strips_parenthetical(self) -> None:
        assert strip_name_decorations("Shohei Ohtani (Batter)") == "Shohei Ohtani"

    def test_no_change_for_plain_name(self) -> None:
        assert strip_name_decorations("Mike Trout") == "Mike Trout"

    def test_strips_both(self) -> None:
        assert strip_name_decorations("Some Player Jr. (Batter)") == "Some Player"
