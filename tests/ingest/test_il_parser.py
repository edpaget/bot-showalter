from fantasy_baseball_manager.ingest.il_parser import ILParseResult, parse_il_transaction


class TestParseILTransaction:
    def test_placement_with_injury(self) -> None:
        desc = "San Diego Padres placed RHP Joe Musgrove on the 15-day injured list. Right elbow inflammation."
        result = parse_il_transaction(desc)
        assert result == ILParseResult(
            transaction_type="placement",
            il_type="15",
            injury_description="Right elbow inflammation",
        )

    def test_placement_retroactive(self) -> None:
        desc = (
            "New York Yankees placed LHP Nestor Cortes on the 15-day injured list"
            " retroactive to May 29, 2024. Left rotator cuff strain."
        )
        result = parse_il_transaction(desc)
        assert result is not None
        assert result.transaction_type == "placement"
        assert result.il_type == "15"
        assert result.injury_description == "Left rotator cuff strain"

    def test_activation(self) -> None:
        desc = "Los Angeles Dodgers activated CF Mookie Betts from the 10-day injured list."
        result = parse_il_transaction(desc)
        assert result == ILParseResult(
            transaction_type="activation",
            il_type="10",
            injury_description=None,
        )

    def test_transfer(self) -> None:
        desc = "Chicago Cubs transferred LHP Justin Steele from the 15-day injured list to the 60-day injured list."
        result = parse_il_transaction(desc)
        assert result is not None
        assert result.transaction_type == "transfer"
        assert result.il_type == "60"

    def test_non_il_status_change_returns_none(self) -> None:
        desc = "Los Angeles Dodgers placed CF Mookie Betts on the paternity list."
        result = parse_il_transaction(desc)
        assert result is None

    def test_no_injury_text_after_placement(self) -> None:
        desc = "San Diego Padres placed RHP Joe Musgrove on the 15-day injured list."
        result = parse_il_transaction(desc)
        assert result is not None
        assert result.transaction_type == "placement"
        assert result.il_type == "15"
        assert result.injury_description is None

    def test_7_day_il(self) -> None:
        desc = "New York Mets placed C Francisco Alvarez on the 7-day injured list. Concussion."
        result = parse_il_transaction(desc)
        assert result is not None
        assert result.il_type == "7"
        assert result.injury_description == "Concussion"

    def test_activation_with_no_match(self) -> None:
        desc = "Kansas City Royals selected the contract of RHP Sam Long from Omaha."
        result = parse_il_transaction(desc)
        assert result is None

    def test_transfer_with_injury(self) -> None:
        desc = (
            "Boston Red Sox transferred SS Trevor Story from the 15-day injured list"
            " to the 60-day injured list. Left elbow surgery."
        )
        result = parse_il_transaction(desc)
        assert result is not None
        assert result.transaction_type == "transfer"
        assert result.il_type == "60"
        assert result.injury_description == "Left elbow surgery"
