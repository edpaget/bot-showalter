import csv
import html
from typing import TextIO

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow, TierAssignment
from fantasy_baseball_manager.domain.league_settings import LeagueSettings
from fantasy_baseball_manager.domain.valuation import Valuation

_PITCHER_POSITIONS = {"SP", "RP"}

_BATTER_POSITION_ORDER = ["C", "1B", "2B", "SS", "3B", "OF", "DH"]
_PITCHER_POSITION_ORDER = ["SP", "RP"]

_TIER_COLORS = [
    "#e8f5e9",  # tier 1: soft green
    "#e3f2fd",  # tier 2: soft blue
    "#fff3e0",  # tier 3: soft orange
    "#fce4ec",  # tier 4: soft pink
    "#f3e5f5",  # tier 5: soft purple
    "#e0f7fa",  # tier 6: soft cyan
    "#fffde7",  # tier 7: soft yellow
    "#efebe9",  # tier 8: soft brown
]


def _is_pitcher_adp(adp: ADP) -> bool:
    return all(p.strip() in _PITCHER_POSITIONS for p in adp.positions.split(",") if p.strip())


def _resolve_adp(entries: list[ADP], is_pitcher: bool) -> ADP:
    matching = [e for e in entries if _is_pitcher_adp(e) == is_pitcher]
    return min(matching or entries, key=lambda a: a.overall_pick)


def build_draft_board(
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str],
    *,
    tiers: list[TierAssignment] | None = None,
    adp: list[ADP] | None = None,
) -> DraftBoard:
    batting_categories = tuple(c.key for c in league.batting_categories)
    pitching_categories = tuple(c.key for c in league.pitching_categories)

    tier_lookup: dict[int, int] = {}
    if tiers is not None:
        tier_lookup = {t.player_id: t.tier for t in tiers}

    adp_by_player: dict[int, list[ADP]] = {}
    if adp is not None:
        for entry in adp:
            adp_by_player.setdefault(entry.player_id, []).append(entry)

    sorted_valuations = sorted(valuations, key=lambda v: v.value, reverse=True)

    rows: list[DraftBoardRow] = []
    for rank, val in enumerate(sorted_valuations, start=1):
        is_pitcher = val.player_type == "pitcher"
        cat_keys = pitching_categories if is_pitcher else batting_categories
        category_z_scores = {k: v for k, v in val.category_scores.items() if k in cat_keys}

        tier = tier_lookup.get(val.player_id)

        adp_overall: float | None = None
        adp_rank: int | None = None
        adp_delta: int | None = None
        if val.player_id in adp_by_player:
            best = _resolve_adp(adp_by_player[val.player_id], is_pitcher)
            adp_overall = best.overall_pick
            adp_rank = best.rank
            adp_delta = best.rank - rank

        player_name = player_names.get(val.player_id, f"Unknown ({val.player_id})")

        rows.append(
            DraftBoardRow(
                player_id=val.player_id,
                player_name=player_name,
                rank=rank,
                player_type=val.player_type,
                position=val.position,
                value=val.value,
                category_z_scores=category_z_scores,
                tier=tier,
                adp_overall=adp_overall,
                adp_rank=adp_rank,
                adp_delta=adp_delta,
            )
        )

    return DraftBoard(
        rows=rows,
        batting_categories=batting_categories,
        pitching_categories=pitching_categories,
    )


def export_csv(board: DraftBoard, output: TextIO) -> None:
    """Write a draft board to CSV format."""
    has_tier = any(r.tier is not None for r in board.rows)
    has_adp = any(r.adp_overall is not None for r in board.rows)

    fieldnames: list[str] = ["Rank", "Player", "Type", "Pos", "Value"]
    if has_tier:
        fieldnames.append("Tier")
    fieldnames.extend(board.batting_categories)
    fieldnames.extend(board.pitching_categories)
    if has_adp:
        fieldnames.extend(["ADP", "ADPRk", "Delta"])

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row in board.rows:
        is_pitcher = row.player_type == "pitcher"
        record: dict[str, str] = {
            "Rank": str(row.rank),
            "Player": row.player_name,
            "Type": row.player_type,
            "Pos": row.position,
            "Value": f"${row.value:.1f}",
        }
        if has_tier:
            record["Tier"] = str(row.tier) if row.tier is not None else ""

        for cat in board.batting_categories:
            if is_pitcher:
                record[cat] = ""
            else:
                z = row.category_z_scores.get(cat)
                record[cat] = f"{z:.2f}" if z is not None else ""

        for cat in board.pitching_categories:
            if not is_pitcher:
                record[cat] = ""
            else:
                z = row.category_z_scores.get(cat)
                record[cat] = f"{z:.2f}" if z is not None else ""

        if has_adp:
            record["ADP"] = f"{row.adp_overall:.1f}" if row.adp_overall is not None else ""
            record["ADPRk"] = str(row.adp_rank) if row.adp_rank is not None else ""
            record["Delta"] = str(row.adp_delta) if row.adp_delta is not None else ""

        writer.writerow(record)


def _position_sort_key(position: str, is_pitcher: bool) -> int:
    order = _PITCHER_POSITION_ORDER if is_pitcher else _BATTER_POSITION_ORDER
    try:
        return order.index(position)
    except ValueError:
        return len(order)


def _group_rows(rows: list[DraftBoardRow]) -> list[tuple[str, list[tuple[str, list[DraftBoardRow]]]]]:
    """Group rows into (type_label, [(position, [rows])]) with correct ordering."""
    batters = [r for r in rows if r.player_type != "pitcher"]
    pitchers = [r for r in rows if r.player_type == "pitcher"]

    groups: list[tuple[str, list[tuple[str, list[DraftBoardRow]]]]] = []
    for label, group_rows, is_pitcher in [("Batters", batters, False), ("Pitchers", pitchers, True)]:
        if not group_rows:
            continue
        by_pos: dict[str, list[DraftBoardRow]] = {}
        for r in group_rows:
            by_pos.setdefault(r.position, []).append(r)
        sorted_positions = sorted(by_pos.keys(), key=lambda p: _position_sort_key(p, is_pitcher))
        pos_groups: list[tuple[str, list[DraftBoardRow]]] = []
        for pos in sorted_positions:
            pos_rows = sorted(by_pos[pos], key=lambda r: r.value, reverse=True)
            pos_groups.append((pos, pos_rows))
        groups.append((label, pos_groups))

    return groups


def export_html(
    board: DraftBoard,
    league: LeagueSettings,
    output: TextIO,
    *,
    adp_delta_threshold: int = 10,
    auto_refresh: int | None = None,
) -> None:
    """Write a draft board to styled HTML format."""
    has_tier = any(r.tier is not None for r in board.rows)
    has_adp = any(r.adp_overall is not None for r in board.rows)
    e = html.escape

    headers: list[str] = ["Rank", "Player", "Pos", "Value"]
    if has_tier:
        headers.append("Tier")
    headers.extend(board.batting_categories)
    headers.extend(board.pitching_categories)
    if has_adp:
        headers.extend(["ADP", "ADPRk", "Delta"])

    col_count = len(headers)

    parts: list[str] = []
    w = parts.append

    w("<!DOCTYPE html>\n")
    w('<html lang="en">\n<head>\n<meta charset="utf-8">\n')
    w(f"<title>{e(league.name)} — Draft Board</title>\n")
    if auto_refresh is not None:
        w(f'<meta http-equiv="refresh" content="{auto_refresh}">\n')
    w("<style>\n")
    w("body { font-family: system-ui, sans-serif; margin: 1em; }\n")
    w("table { border-collapse: collapse; width: 100%; font-size: 0.85em; }\n")
    w("th, td { border: 1px solid #ccc; padding: 3px 6px; text-align: left; }\n")
    w("th { background: #f5f5f5; position: sticky; top: 0; }\n")
    w(".buy { color: #006600; font-weight: bold; }\n")
    w(".avoid { color: #cc0000; font-weight: bold; }\n")
    w(".type-header td { background: #333; color: #fff; font-weight: bold; font-size: 1.1em; }\n")
    w(".pos-header td { background: #e0e0e0; font-weight: bold; }\n")
    for i, color in enumerate(_TIER_COLORS, start=1):
        w(f".tier-{i} {{ background: {color}; }}\n")
    w("@media print {\n")
    w("  @page { size: landscape; margin: 0.5cm; }\n")
    w("  body { font-size: 8pt; margin: 0; }\n")
    w("  tr { page-break-inside: avoid; }\n")
    w("  th { position: static; }\n")
    w("}\n")
    w("</style>\n")
    w("</head>\n<body>\n")
    w(f"<h1>{e(league.name)} — Draft Board</h1>\n")
    w("<table>\n<thead>\n<tr>")
    for h in headers:
        w(f"<th>{e(h)}</th>")
    w("</tr>\n</thead>\n<tbody>\n")

    grouped = _group_rows(board.rows)
    for type_label, pos_groups in grouped:
        w(f'<tr class="type-header"><td colspan="{col_count}">{e(type_label)}</td></tr>\n')
        for pos_label, pos_rows in pos_groups:
            w(f'<tr class="pos-header"><td colspan="{col_count}">{e(pos_label)}</td></tr>\n')
            for row in pos_rows:
                tier_class = ""
                if row.tier is not None:
                    tier_css_idx = ((row.tier - 1) % 8) + 1
                    tier_class = f' class="tier-{tier_css_idx}"'
                w(f"<tr{tier_class}>")

                is_pitcher = row.player_type == "pitcher"
                w(f"<td>{row.rank}</td>")
                w(f"<td>{e(row.player_name)}</td>")
                w(f"<td>{e(row.position)}</td>")
                w(f"<td>${row.value:.1f}</td>")

                if has_tier:
                    w(f"<td>{row.tier if row.tier is not None else ''}</td>")

                for cat in board.batting_categories:
                    if is_pitcher:
                        w("<td></td>")
                    else:
                        z = row.category_z_scores.get(cat)
                        w(f"<td>{z:.2f}</td>" if z is not None else "<td></td>")

                for cat in board.pitching_categories:
                    if not is_pitcher:
                        w("<td></td>")
                    else:
                        z = row.category_z_scores.get(cat)
                        w(f"<td>{z:.2f}</td>" if z is not None else "<td></td>")

                if has_adp:
                    w(f"<td>{row.adp_overall:.1f}</td>" if row.adp_overall is not None else "<td></td>")
                    w(f"<td>{row.adp_rank}</td>" if row.adp_rank is not None else "<td></td>")
                    if row.adp_delta is not None:
                        delta_class = ""
                        if row.adp_delta >= adp_delta_threshold:
                            delta_class = ' class="buy"'
                        elif row.adp_delta <= -adp_delta_threshold:
                            delta_class = ' class="avoid"'
                        w(f"<td{delta_class}>{row.adp_delta}</td>")
                    else:
                        w("<td></td>")

                w("</tr>\n")

    w("</tbody>\n</table>\n</body>\n</html>")

    output.write("".join(parts))
