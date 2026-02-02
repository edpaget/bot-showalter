from fantasy_baseball_manager.pipeline.types import PlayerRates


class TestPlayerRates:
    def test_construction_with_defaults(self) -> None:
        pr = PlayerRates(player_id="p1", name="Test", year=2025, age=28)
        assert pr.player_id == "p1"
        assert pr.name == "Test"
        assert pr.year == 2025
        assert pr.age == 28
        assert pr.rates == {}
        assert pr.opportunities == 0.0
        assert pr.metadata == {}

    def test_construction_with_rates(self) -> None:
        rates = {"hr": 0.04, "bb": 0.08}
        pr = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=28,
            rates=rates,
            opportunities=555.0,
        )
        assert pr.rates == {"hr": 0.04, "bb": 0.08}
        assert pr.opportunities == 555.0

    def test_mutable_rates(self) -> None:
        pr = PlayerRates(player_id="p1", name="Test", year=2025, age=28)
        pr.rates["hr"] = 0.04
        assert pr.rates["hr"] == 0.04

    def test_metadata_carries_arbitrary_data(self) -> None:
        pr = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=28,
            metadata={"pa_per_year": [600, 550, 500], "is_starter": True},
        )
        assert pr.metadata["pa_per_year"] == [600, 550, 500]
        assert pr.metadata["is_starter"] is True

    def test_instances_do_not_share_defaults(self) -> None:
        pr1 = PlayerRates(player_id="p1", name="A", year=2025, age=28)
        pr2 = PlayerRates(player_id="p2", name="B", year=2025, age=30)
        pr1.rates["hr"] = 0.04
        assert "hr" not in pr2.rates
