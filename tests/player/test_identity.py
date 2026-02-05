"""Tests for the player identity module."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.player.identity import Player


class TestPlayer:
    """Tests for the Player dataclass."""

    def test_required_fields(self) -> None:
        """Player requires name and yahoo_id."""
        player = Player(name="Mike Trout", yahoo_id="10155")

        assert player.name == "Mike Trout"
        assert player.yahoo_id == "10155"

    def test_default_values(self) -> None:
        """Optional fields have sensible defaults."""
        player = Player(name="Test Player", yahoo_id="123")

        assert player.yahoo_sub_id is None
        assert player.fangraphs_id is None
        assert player.mlbam_id is None
        assert player.team is None
        assert player.eligible_positions == ()
        assert player.age is None

    def test_frozen(self) -> None:
        """Player is immutable."""
        player = Player(name="Test", yahoo_id="123")
        with pytest.raises(AttributeError):
            player.name = "Changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Players with same attributes are equal."""
        player1 = Player(name="Test", yahoo_id="123")
        player2 = Player(name="Test", yahoo_id="123")
        assert player1 == player2

    def test_hashable(self) -> None:
        """Players can be used as dict keys."""
        player = Player(name="Test", yahoo_id="123")
        d = {player: "value"}
        assert d[player] == "value"


class TestIsTwoWay:
    """Tests for the is_two_way property."""

    def test_regular_player_not_two_way(self) -> None:
        """Regular players are not two-way."""
        player = Player(name="Mike Trout", yahoo_id="10155")
        assert player.is_two_way is False

    def test_player_with_sub_id_is_two_way(self) -> None:
        """Players with yahoo_sub_id are two-way."""
        ohtani_batter = Player(
            name="Shohei Ohtani",
            yahoo_id="10835",
            yahoo_sub_id="1000001",
        )
        assert ohtani_batter.is_two_way is True


class TestEffectiveYahooId:
    """Tests for the effective_yahoo_id property."""

    def test_regular_player_uses_yahoo_id(self) -> None:
        """Regular players use yahoo_id."""
        player = Player(name="Mike Trout", yahoo_id="10155")
        assert player.effective_yahoo_id == "10155"

    def test_two_way_player_uses_sub_id(self) -> None:
        """Two-way players use yahoo_sub_id."""
        ohtani_batter = Player(
            name="Shohei Ohtani",
            yahoo_id="10835",
            yahoo_sub_id="1000001",
        )
        assert ohtani_batter.effective_yahoo_id == "1000001"


class TestWithIds:
    """Tests for the with_ids method."""

    def test_adds_fangraphs_id(self) -> None:
        """with_ids() adds FanGraphs ID."""
        player = Player(name="Test", yahoo_id="123")
        enriched = player.with_ids(fangraphs_id="fg456")

        assert enriched.fangraphs_id == "fg456"
        assert enriched.yahoo_id == "123"  # Preserved
        assert enriched.name == "Test"  # Preserved

    def test_adds_mlbam_id(self) -> None:
        """with_ids() adds MLBAM ID."""
        player = Player(name="Test", yahoo_id="123")
        enriched = player.with_ids(mlbam_id="mlb789")

        assert enriched.mlbam_id == "mlb789"

    def test_adds_both_ids(self) -> None:
        """with_ids() can add both IDs at once."""
        player = Player(name="Test", yahoo_id="123")
        enriched = player.with_ids(fangraphs_id="fg456", mlbam_id="mlb789")

        assert enriched.fangraphs_id == "fg456"
        assert enriched.mlbam_id == "mlb789"

    def test_preserves_existing_ids(self) -> None:
        """with_ids() preserves existing IDs if not overridden."""
        player = Player(name="Test", yahoo_id="123", fangraphs_id="existing")
        enriched = player.with_ids(mlbam_id="mlb789")

        assert enriched.fangraphs_id == "existing"  # Preserved
        assert enriched.mlbam_id == "mlb789"  # Added

    def test_returns_new_instance(self) -> None:
        """with_ids() returns a new Player instance."""
        player = Player(name="Test", yahoo_id="123")
        enriched = player.with_ids(fangraphs_id="fg456")

        assert player is not enriched
        assert player.fangraphs_id is None  # Original unchanged


class TestWithMetadata:
    """Tests for the with_metadata method."""

    def test_adds_team(self) -> None:
        """with_metadata() adds team."""
        player = Player(name="Test", yahoo_id="123")
        enriched = player.with_metadata(team="LAA")

        assert enriched.team == "LAA"

    def test_adds_positions(self) -> None:
        """with_metadata() adds eligible positions."""
        player = Player(name="Test", yahoo_id="123")
        enriched = player.with_metadata(eligible_positions=("C", "1B", "DH"))

        assert enriched.eligible_positions == ("C", "1B", "DH")

    def test_adds_age(self) -> None:
        """with_metadata() adds age."""
        player = Player(name="Test", yahoo_id="123")
        enriched = player.with_metadata(age=32)

        assert enriched.age == 32

    def test_returns_new_instance(self) -> None:
        """with_metadata() returns a new Player instance."""
        player = Player(name="Test", yahoo_id="123")
        enriched = player.with_metadata(team="LAA")

        assert player is not enriched
        assert player.team is None  # Original unchanged


class TestTwoWayPlayerUsage:
    """Tests demonstrating two-way player handling."""

    def test_ohtani_batter_and_pitcher_are_different(self) -> None:
        """Ohtani as batter and pitcher are distinct Player objects."""
        ohtani_batter = Player(
            name="Shohei Ohtani",
            yahoo_id="10835",
            yahoo_sub_id="1000001",
            fangraphs_id="19755",
        )
        ohtani_pitcher = Player(
            name="Shohei Ohtani",
            yahoo_id="10835",
            yahoo_sub_id="1000002",
            fangraphs_id="19755",
        )

        # Same base player
        assert ohtani_batter.yahoo_id == ohtani_pitcher.yahoo_id
        assert ohtani_batter.fangraphs_id == ohtani_pitcher.fangraphs_id

        # Different effective IDs
        assert ohtani_batter.effective_yahoo_id != ohtani_pitcher.effective_yahoo_id

        # Not equal (different sub_id)
        assert ohtani_batter != ohtani_pitcher

    def test_can_use_as_dict_keys(self) -> None:
        """Two-way player representations can be used as separate dict keys."""
        ohtani_batter = Player(
            name="Shohei Ohtani",
            yahoo_id="10835",
            yahoo_sub_id="1000001",
        )
        ohtani_pitcher = Player(
            name="Shohei Ohtani",
            yahoo_id="10835",
            yahoo_sub_id="1000002",
        )

        stats = {
            ohtani_batter: {"HR": 40, "AVG": 0.285},
            ohtani_pitcher: {"W": 12, "ERA": 3.01},
        }

        assert stats[ohtani_batter]["HR"] == 40
        assert stats[ohtani_pitcher]["W"] == 12
