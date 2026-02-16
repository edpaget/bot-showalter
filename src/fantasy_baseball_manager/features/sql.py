from __future__ import annotations

from dataclasses import dataclass

from fantasy_baseball_manager.features.types import (
    AnyFeature,
    DeltaFeature,
    DerivedTransformFeature,
    Feature,
    FeatureSet,
    Source,
    TransformFeature,
)

_SOURCE_TABLES: dict[Source, str] = {
    Source.BATTING: "batting_stats",
    Source.PITCHING: "pitching_stats",
    Source.PLAYER: "player",
    Source.PROJECTION: "projection",
    Source.IL_STINT: "il_stint",
}


_ALIAS_PREFIXES: dict[Source, str] = {
    Source.BATTING: "b",
    Source.PITCHING: "pi",
    Source.PLAYER: "p",
    Source.PROJECTION: "pr",
    Source.IL_STINT: "ils",
}


def _source_table(source: Source) -> str:
    return _SOURCE_TABLES[source]


@dataclass(frozen=True)
class JoinSpec:
    source: Source
    lag: int
    alias: str
    table: str
    system: str | None = None
    version: str | None = None
    distribution_stat: str | None = None
    projection_alias: str | None = None


def _join_alias(source: Source, lag: int, counter: int | None = None) -> str:
    prefix = _ALIAS_PREFIXES[source]
    if source == Source.PLAYER:
        return prefix
    if counter is not None:
        return f"{prefix}{counter}"
    return f"{prefix}{lag}"


def _is_rolling(feature: Feature) -> bool:
    return feature.window > 1 and feature.aggregate is not None


_AGG_FUNCTIONS: dict[str, str] = {
    "mean": "AVG",
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
}


def _extract_features(features: tuple[AnyFeature, ...]) -> list[Feature]:
    """Extract all plain Features from a tuple that may contain DeltaFeatures.

    TransformFeature instances are skipped — they are handled in the
    transform pass, not SQL generation.
    """
    result: list[Feature] = []
    for f in features:
        if isinstance(f, TransformFeature):
            continue
        if isinstance(f, DerivedTransformFeature):
            continue
        if isinstance(f, DeltaFeature):
            result.append(f.left)
            result.append(f.right)
        else:
            result.append(f)
    return result


def _plan_joins(features: tuple[AnyFeature, ...]) -> list[JoinSpec]:
    plain_features = _extract_features(features)
    seen: dict[tuple[Source, int, str | None, str | None], None] = {}
    counter = 0
    for f in plain_features:
        if _is_rolling(f):
            continue
        lag = 0 if f.source == Source.PLAYER else f.lag
        key = (f.source, lag, f.system, f.version)
        if key not in seen:
            seen[key] = None
    joins: list[JoinSpec] = []
    for source, lag, system, version in sorted(seen, key=lambda k: (k[0].value, k[1], k[2] or "", k[3] or "")):
        if source == Source.PROJECTION:
            alias = _join_alias(source, lag, counter=counter)
            counter += 1
        else:
            alias = _join_alias(source, lag)
        joins.append(
            JoinSpec(
                source=source,
                lag=lag,
                alias=alias,
                table=_source_table(source),
                system=system,
                version=version,
            )
        )

    # Plan distribution joins for features with distribution_column set
    # Dedup key: (source, lag, system, version, stat) where stat = column
    proj_lookup = {(j.source, j.lag, j.system, j.version): j for j in joins}
    dist_seen: dict[tuple[Source, int, str | None, str | None, str], None] = {}
    for f in plain_features:
        if f.distribution_column is None:
            continue
        lag = 0 if f.source == Source.PLAYER else f.lag
        dist_key = (f.source, lag, f.system, f.version, f.column)
        if dist_key not in dist_seen:
            dist_seen[dist_key] = None

    pd_counter = 0
    for source, lag, system, version, stat in sorted(
        dist_seen, key=lambda k: (k[0].value, k[1], k[2] or "", k[3] or "", k[4])
    ):
        proj_join = proj_lookup[(source, lag, system, version)]
        joins.append(
            JoinSpec(
                source=source,
                lag=lag,
                alias=f"pd{pd_counter}",
                table="projection_distribution",
                system=system,
                version=version,
                distribution_stat=stat,
                projection_alias=proj_join.alias,
            )
        )
        pd_counter += 1

    return joins


def _rolling_subquery(
    col: str,
    agg: str,
    table: str,
    lag: int,
    window: int,
    source_filter: str | None,
) -> tuple[str, list[object]]:
    lower = lag + window - 1
    upper = lag
    func = _AGG_FUNCTIONS[agg]
    parts = [
        f"(SELECT {func}([{col}]) FROM {table}",
        " WHERE player_id = spine.player_id",
        f" AND season BETWEEN spine.season - {lower} AND spine.season - {upper}",
    ]
    params: list[object] = []
    if source_filter is not None and table != "il_stint":
        parts.append(" AND source = ?")
        params.append(source_filter)
    parts.append(")")
    return "".join(parts), params


def _raw_expr(
    feature: Feature,
    joins_dict: dict[tuple[Source, int, str | None, str | None], JoinSpec],
    source_filter: str | None,
    *,
    dist_joins_dict: dict[tuple[Source, int, str | None, str | None, str], JoinSpec] | None = None,
) -> tuple[str, list[object]]:
    """Return the raw SQL expression for a feature (without AS alias)."""
    # Distribution column (e.g., p90, std)
    if feature.distribution_column is not None and dist_joins_dict is not None:
        lag = 0 if feature.source == Source.PLAYER else feature.lag
        dist_key = (feature.source, lag, feature.system, feature.version, feature.column)
        dist_join = dist_joins_dict[dist_key]
        return f"{dist_join.alias}.[{feature.distribution_column}]", []

    # Computed age
    if feature.computed == "age":
        alias = joins_dict[(Source.PLAYER, 0, None, None)].alias
        return (
            f"spine.season - CAST(SUBSTR({alias}.birth_date, 1, 4) AS INTEGER)",
            [],
        )

    # Computed positions
    if feature.computed == "positions":
        return (
            "(SELECT GROUP_CONCAT(pa.position, ',')"
            " FROM (SELECT position, games FROM position_appearance"
            " WHERE player_id = spine.player_id"
            " AND season = spine.season - 1"
            " ORDER BY games DESC) pa)",
            [],
        )

    table = _source_table(feature.source)
    lag = 0 if feature.source == Source.PLAYER else feature.lag

    # Rolling rate (aggregate + denominator)
    if _is_rolling(feature) and feature.denominator is not None:
        num_sql, num_params = _rolling_subquery(feature.column, "sum", table, lag, feature.window, source_filter)
        den_sql, den_params = _rolling_subquery(feature.denominator, "sum", table, lag, feature.window, source_filter)
        return (
            f"CAST({num_sql} AS REAL) / NULLIF({den_sql}, 0)",
            num_params + den_params,
        )

    # Rolling aggregate (no denominator)
    if _is_rolling(feature) and feature.aggregate is not None:
        sub_sql, sub_params = _rolling_subquery(
            feature.column, feature.aggregate, table, lag, feature.window, source_filter
        )
        return sub_sql, sub_params

    # Non-rolling: need alias from joins_dict
    key = (feature.source, lag, feature.system, feature.version)
    alias = joins_dict[key].alias

    # Rate stat (denominator, no aggregate)
    if feature.denominator is not None:
        return (
            f"CAST({alias}.[{feature.column}] AS REAL) / NULLIF({alias}.[{feature.denominator}], 0)",
            [],
        )

    # Direct column
    return f"{alias}.[{feature.column}]", []


def _select_expr(
    feature: AnyFeature,
    joins_dict: dict[tuple[Source, int, str | None, str | None], JoinSpec],
    source_filter: str | None,
    *,
    dist_joins_dict: dict[tuple[Source, int, str | None, str | None, str], JoinSpec] | None = None,
) -> tuple[str, list[object]]:
    if isinstance(feature, DeltaFeature):
        left_sql, left_params = _raw_expr(feature.left, joins_dict, source_filter, dist_joins_dict=dist_joins_dict)
        right_sql, right_params = _raw_expr(feature.right, joins_dict, source_filter, dist_joins_dict=dist_joins_dict)
        return f"({left_sql} - {right_sql}) AS [{feature.name}]", left_params + right_params

    assert isinstance(feature, Feature)
    expr, params = _raw_expr(feature, joins_dict, source_filter, dist_joins_dict=dist_joins_dict)
    return f"{expr} AS [{feature.name}]", params


def _infer_spine_table(feature_set: FeatureSet) -> str:
    sf = feature_set.spine_filter
    if sf.player_type == "batter":
        return "batting_stats"
    if sf.player_type == "pitcher":
        return "pitching_stats"
    for f in _extract_features(feature_set.features):
        if f.source in (Source.BATTING, Source.PITCHING):
            return _source_table(f.source)
    return "batting_stats"


def _spine_cte(feature_set: FeatureSet) -> tuple[str, list[object]]:
    table = _infer_spine_table(feature_set)
    placeholders = ", ".join("?" for _ in feature_set.seasons)
    parts = [
        f"SELECT DISTINCT player_id, season FROM {table}",
        f" WHERE season IN ({placeholders})",
    ]
    params: list[object] = list(feature_set.seasons)

    if feature_set.source_filter is not None:
        parts.append(" AND source = ?")
        params.append(feature_set.source_filter)

    sf = feature_set.spine_filter
    if sf.min_pa is not None:
        parts.append(" AND pa >= ?")
        params.append(sf.min_pa)
    if sf.min_ip is not None:
        parts.append(" AND ip >= ?")
        params.append(sf.min_ip)

    return "".join(parts), params


def _join_clause(
    join: JoinSpec, source_filter: str | None, *, player_type: str | None = None
) -> tuple[str, list[object]]:
    alias = join.alias

    # Distribution join
    if join.distribution_stat is not None:
        parts = [
            f"LEFT JOIN {join.table} {alias}",
            f" ON {alias}.projection_id = {join.projection_alias}.id",
            f" AND {alias}.stat = ?",
        ]
        return "".join(parts), [join.distribution_stat]

    if join.source == Source.PLAYER:
        return f"LEFT JOIN {join.table} {alias} ON {alias}.id = spine.player_id", []

    if join.source == Source.IL_STINT:
        season_expr = "spine.season" if join.lag == 0 else f"spine.season - {join.lag}"
        subquery = (
            "(SELECT player_id, season, COALESCE(SUM(days), 0) AS days, COUNT(*) AS stint_count"
            f" FROM {join.table} GROUP BY player_id, season)"
        )
        return (
            f"LEFT JOIN {subquery} {alias} ON {alias}.player_id = spine.player_id AND {alias}.season = {season_expr}",
            [],
        )

    season_expr = "spine.season" if join.lag == 0 else f"spine.season - {join.lag}"
    parts = [
        f"LEFT JOIN {join.table} {alias}",
        f" ON {alias}.player_id = spine.player_id",
        f" AND {alias}.season = {season_expr}",
    ]
    params: list[object] = []

    if join.source == Source.PROJECTION:
        # Projection joins use system filter instead of source filter
        if join.system is not None:
            parts.append(f" AND {alias}.system = ?")
            params.append(join.system)
        if join.version is not None:
            parts.append(f" AND {alias}.version = ?")
            params.append(join.version)
        if player_type is not None:
            parts.append(f" AND {alias}.player_type = ?")
            params.append(player_type)
    elif source_filter is not None:
        parts.append(f" AND {alias}.source = ?")
        params.append(source_filter)

    return "".join(parts), params


def generate_sql(feature_set: FeatureSet) -> tuple[str, list[object]]:
    """Return (sql_string, params) for a parameterized SELECT query."""
    # Build spine CTE
    spine_sql, spine_params = _spine_cte(feature_set)

    # Plan joins
    joins = _plan_joins(feature_set.features)
    joins_dict = {(j.source, j.lag, j.system, j.version): j for j in joins if j.distribution_stat is None}
    dist_joins_dict = {
        (j.source, j.lag, j.system, j.version, j.distribution_stat): j for j in joins if j.distribution_stat is not None
    }

    # Build SELECT expressions (skip TransformFeature — handled in transform pass)
    select_parts = ["spine.player_id", "spine.season"]
    select_params: list[object] = []
    for f in feature_set.features:
        if isinstance(f, (TransformFeature, DerivedTransformFeature)):
            continue
        expr, expr_params = _select_expr(f, joins_dict, feature_set.source_filter, dist_joins_dict=dist_joins_dict)
        select_parts.append(expr)
        select_params.extend(expr_params)

    # Build JOIN clauses
    join_parts: list[str] = []
    join_params: list[object] = []
    for j in joins:
        clause, clause_params = _join_clause(
            j, feature_set.source_filter, player_type=feature_set.spine_filter.player_type
        )
        join_parts.append(clause)
        join_params.extend(clause_params)

    # Assemble query
    select_list = ",\n    ".join(select_parts)
    sql_parts = [
        f"WITH spine AS (\n    {spine_sql}\n)\n",
        f"SELECT\n    {select_list}\n",
        "FROM spine\n",
    ]
    for jp in join_parts:
        sql_parts.append(f"{jp}\n")

    sql = "".join(sql_parts).rstrip("\n")
    params = spine_params + select_params + join_params
    return sql, params
