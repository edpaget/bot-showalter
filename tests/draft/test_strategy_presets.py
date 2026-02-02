from fantasy_baseball_manager.draft.simulation_models import DraftRule, DraftStrategy
from fantasy_baseball_manager.draft.strategy_presets import STRATEGY_PRESETS
from fantasy_baseball_manager.valuation.models import StatCategory


class TestStrategyPresets:
    def test_all_presets_are_draft_strategies(self) -> None:
        for name, strategy in STRATEGY_PRESETS.items():
            assert isinstance(strategy, DraftStrategy), f"{name} is not a DraftStrategy"

    def test_expected_presets_exist(self) -> None:
        expected = {"balanced", "power_hitting", "speed", "pitching_heavy", "punt_saves"}
        assert expected == set(STRATEGY_PRESETS.keys())

    def test_preset_names_match_keys(self) -> None:
        for key, strategy in STRATEGY_PRESETS.items():
            assert strategy.name == key

    def test_balanced_has_default_weights(self) -> None:
        s = STRATEGY_PRESETS["balanced"]
        assert s.category_weights == {}

    def test_power_hitting_has_hr_weight(self) -> None:
        s = STRATEGY_PRESETS["power_hitting"]
        assert s.category_weights.get(StatCategory.HR, 1.0) > 1.0

    def test_speed_has_sb_weight(self) -> None:
        s = STRATEGY_PRESETS["speed"]
        assert s.category_weights.get(StatCategory.SB, 1.0) > 1.0

    def test_pitching_heavy_has_pitching_weights(self) -> None:
        s = STRATEGY_PRESETS["pitching_heavy"]
        assert s.category_weights.get(StatCategory.K, 1.0) > 1.0
        assert s.category_weights.get(StatCategory.ERA, 1.0) > 1.0

    def test_punt_saves_zeroes_nsvh(self) -> None:
        s = STRATEGY_PRESETS["punt_saves"]
        assert s.category_weights.get(StatCategory.NSVH) == 0.0

    def test_all_presets_have_rules(self) -> None:
        for name, strategy in STRATEGY_PRESETS.items():
            assert isinstance(strategy.rules, tuple), f"{name} rules is not a tuple"
            for rule in strategy.rules:
                assert isinstance(rule, DraftRule), f"{name} has a non-DraftRule rule"

    def test_all_presets_have_valid_noise_scale(self) -> None:
        for name, strategy in STRATEGY_PRESETS.items():
            assert strategy.noise_scale >= 0, f"{name} has negative noise_scale"
