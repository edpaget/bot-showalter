import pytest

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.models.ensemble.stat_groups import (
    BUILTIN_GROUPS,
    expand_route_groups,
    league_required_stats,
    validate_coverage,
)


class TestBuiltinGroups:
    def test_builtin_groups_keys(self) -> None:
        assert set(BUILTIN_GROUPS.keys()) == {
            "batting_counting",
            "batting_rate",
            "pitching_counting",
            "pitching_rate",
            "war",
        }

    def test_builtin_groups_are_frozensets(self) -> None:
        for group in BUILTIN_GROUPS.values():
            assert isinstance(group, frozenset)

    def test_batting_counting_contains_core_stats(self) -> None:
        core = {"hr", "rbi", "r", "sb", "pa", "bb", "so"}
        assert core <= BUILTIN_GROUPS["batting_counting"]

    def test_batting_rate_contains_core_stats(self) -> None:
        core = {"avg", "obp", "slg", "ops", "woba", "iso", "babip"}
        assert core <= BUILTIN_GROUPS["batting_rate"]

    def test_pitching_counting_contains_core_stats(self) -> None:
        core = {"w", "sv", "hld", "ip", "so", "er"}
        assert core <= BUILTIN_GROUPS["pitching_counting"]

    def test_pitching_rate_contains_core_stats(self) -> None:
        core = {"era", "whip", "fip", "k_per_9", "bb_per_9"}
        assert core <= BUILTIN_GROUPS["pitching_rate"]

    def test_war_group(self) -> None:
        assert BUILTIN_GROUPS["war"] == frozenset({"war"})

    def test_no_overlap_counting_and_rate(self) -> None:
        assert BUILTIN_GROUPS["batting_counting"].isdisjoint(BUILTIN_GROUPS["batting_rate"])
        assert BUILTIN_GROUPS["pitching_counting"].isdisjoint(BUILTIN_GROUPS["pitching_rate"])


class TestExpandRouteGroups:
    def test_expand_single_group(self) -> None:
        result = expand_route_groups(route_groups={"batting_rate": "statcast-gbm"})
        for stat in BUILTIN_GROUPS["batting_rate"]:
            assert result[stat] == "statcast-gbm"

    def test_expand_multiple_groups(self) -> None:
        result = expand_route_groups(route_groups={"batting_rate": "statcast-gbm", "batting_counting": "steamer"})
        for stat in BUILTIN_GROUPS["batting_rate"]:
            assert result[stat] == "statcast-gbm"
        for stat in BUILTIN_GROUPS["batting_counting"]:
            assert result[stat] == "steamer"

    def test_per_stat_routes_override_groups(self) -> None:
        result = expand_route_groups(
            route_groups={"batting_rate": "gbm"},
            routes={"obp": "zips"},
        )
        assert result["obp"] == "zips"
        assert result["avg"] == "gbm"

    def test_custom_group(self) -> None:
        result = expand_route_groups(
            route_groups={"my_stats": "gbm"},
            custom_groups={"my_stats": ["hr", "rbi"]},
        )
        assert result["hr"] == "gbm"
        assert result["rbi"] == "gbm"
        assert len(result) == 2

    def test_custom_group_overrides_builtin(self) -> None:
        result = expand_route_groups(
            route_groups={"batting_rate": "gbm"},
            custom_groups={"batting_rate": ["obp", "avg"]},
        )
        assert result == {"obp": "gbm", "avg": "gbm"}

    def test_unknown_group_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stat group"):
            expand_route_groups(route_groups={"nonexistent": "gbm"})

    def test_empty_route_groups(self) -> None:
        result = expand_route_groups(route_groups={})
        assert result == {}

    def test_routes_only_no_groups(self) -> None:
        result = expand_route_groups(
            route_groups={},
            routes={"hr": "steamer", "obp": "gbm"},
        )
        assert result == {"hr": "steamer", "obp": "gbm"}

    def test_league_required_pseudo_group(self) -> None:
        league = _make_league(
            batting=[_counting_cat("hr")],
            pitching=[_rate_cat("era", "er", "ip")],
        )
        result = expand_route_groups(
            route_groups={"league_required": "steamer"},
            league=league,
        )
        assert result["hr"] == "steamer"
        assert result["era"] == "steamer"
        assert result["er"] == "steamer"
        assert result["ip"] == "steamer"

    def test_league_required_with_other_groups(self) -> None:
        league = _make_league(
            batting=[_counting_cat("hr"), _rate_cat("obp", "h+bb+hbp", "pa")],
            pitching=[],
        )
        result = expand_route_groups(
            route_groups={"league_required": "steamer", "batting_rate": "gbm"},
            league=league,
        )
        # obp is in both league_required and batting_rate — batting_rate overrides
        assert result["obp"] == "gbm"
        # hr is only in league_required
        assert result["hr"] == "steamer"

    def test_league_required_without_league_raises(self) -> None:
        with pytest.raises(ValueError, match="league_required.*requires"):
            expand_route_groups(route_groups={"league_required": "steamer"})


def _counting_cat(key: str) -> CategoryConfig:
    return CategoryConfig(key=key, name=key, stat_type=StatType.COUNTING, direction=Direction.HIGHER)


def _rate_cat(key: str, numerator: str, denominator: str) -> CategoryConfig:
    return CategoryConfig(
        key=key,
        name=key,
        stat_type=StatType.RATE,
        direction=Direction.HIGHER,
        numerator=numerator,
        denominator=denominator,
    )


def _make_league(
    batting: list[CategoryConfig],
    pitching: list[CategoryConfig],
) -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=9,
        roster_pitchers=8,
        batting_categories=tuple(batting),
        pitching_categories=tuple(pitching),
    )


class TestValidateCoverage:
    def test_all_stats_covered(self) -> None:
        league = _make_league(
            batting=[_counting_cat("hr"), _counting_cat("rbi")],
            pitching=[],
        )
        routes = {"hr": "steamer", "rbi": "steamer"}
        result = validate_coverage(routes, league)
        assert result == []

    def test_missing_stat(self) -> None:
        league = _make_league(
            batting=[_counting_cat("hr"), _counting_cat("rbi")],
            pitching=[],
        )
        routes = {"hr": "steamer"}
        result = validate_coverage(routes, league)
        assert result == ["rbi"]

    def test_missing_rate_component(self) -> None:
        league = _make_league(
            batting=[_rate_cat("obp", "h+bb+hbp", "pa")],
            pitching=[],
        )
        # Route the key but not all components
        routes = {"obp": "gbm", "h": "steamer"}
        result = validate_coverage(routes, league)
        # bb, hbp, pa are missing
        assert result == ["bb", "hbp", "pa"]

    def test_no_routes(self) -> None:
        league = _make_league(
            batting=[_counting_cat("hr")],
            pitching=[_counting_cat("so")],
        )
        result = validate_coverage({}, league)
        assert result == ["hr", "so"]

    def test_compound_key_components(self) -> None:
        league = _make_league(
            batting=[],
            pitching=[_counting_cat("sv+hld")],
        )
        routes = {"sv": "steamer"}
        result = validate_coverage(routes, league)
        assert result == ["hld"]


class TestLeagueRequiredStats:
    def test_counting_stat_key(self) -> None:
        league = _make_league(batting=[_counting_cat("hr")], pitching=[])
        result = league_required_stats(league)
        assert "hr" in result

    def test_rate_category_key_and_components(self) -> None:
        league = _make_league(
            batting=[_rate_cat("obp", "h+bb+hbp", "pa")],
            pitching=[],
        )
        result = league_required_stats(league)
        assert {"obp", "h", "bb", "hbp", "pa"} <= result

    def test_compound_counting_key(self) -> None:
        league = _make_league(
            batting=[],
            pitching=[_counting_cat("sv+hld")],
        )
        result = league_required_stats(league)
        assert {"sv", "hld"} <= result

    def test_pitching_rate_components(self) -> None:
        league = _make_league(
            batting=[],
            pitching=[_rate_cat("era", "er", "ip")],
        )
        result = league_required_stats(league)
        assert {"era", "er", "ip"} <= result

    def test_full_h2h_league(self) -> None:
        league = _make_league(
            batting=[
                _counting_cat("hr"),
                _counting_cat("r"),
                _counting_cat("rbi"),
                _rate_cat("obp", "h+bb+hbp", "pa"),
                _counting_cat("sb"),
            ],
            pitching=[
                _rate_cat("era", "er", "ip"),
                _rate_cat("whip", "bb+h", "ip"),
                _counting_cat("so"),
                _counting_cat("w"),
                _counting_cat("sv+hld"),
            ],
        )
        result = league_required_stats(league)
        expected = {
            "hr",
            "r",
            "rbi",
            "obp",
            "h",
            "bb",
            "hbp",
            "pa",
            "sb",
            "era",
            "er",
            "ip",
            "whip",
            "so",
            "w",
            "sv",
            "hld",
        }
        assert expected <= result
