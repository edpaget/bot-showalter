import { useState, useMemo } from "react";
import { useQuery } from "@apollo/client";
import { BOARD_QUERY } from "../graphql/queries";
import type { DraftBoardRow } from "../types/board";
import { TIER_COLORS, ADP_DELTA_THRESHOLD } from "../constants/tiers";
import { FilterBar, type PlayerTypeFilter } from "./FilterBar";
import { SearchInput } from "./SearchInput";

type SortKey = keyof Pick<
  DraftBoardRow,
  | "rank"
  | "playerName"
  | "position"
  | "tier"
  | "value"
  | "adpOverall"
  | "adpDelta"
  | "breakoutRank"
  | "bustRank"
>;

type SortDir = "asc" | "desc";

interface BoardData {
  board: {
    rows: DraftBoardRow[];
    battingCategories: string[];
    pitchingCategories: string[];
  };
}

interface BoardVars {
  season: number;
  system?: string;
  version?: string;
}

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: "rank", label: "Rank" },
  { key: "playerName", label: "Player" },
  { key: "position", label: "Pos" },
  { key: "tier", label: "Tier" },
  { key: "value", label: "Value" },
  { key: "adpOverall", label: "ADP" },
  { key: "adpDelta", label: "Delta" },
  { key: "breakoutRank", label: "Breakout" },
  { key: "bustRank", label: "Bust" },
];

function compareValues(
  a: string | number | null | undefined,
  b: string | number | null | undefined,
  dir: SortDir,
): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  const cmp = typeof a === "string" ? a.localeCompare(b as string) : (a as number) - (b as number);
  return dir === "asc" ? cmp : -cmp;
}

function tierBackground(tier: number | null): string | undefined {
  if (tier == null) return undefined;
  return TIER_COLORS[((tier - 1) % TIER_COLORS.length)]!;
}

function breakoutTint(row: DraftBoardRow): string | undefined {
  if (row.breakoutRank != null && row.breakoutRank <= 20) return "rgba(0, 128, 0, 0.06)";
  if (row.bustRank != null && row.bustRank <= 20) return "rgba(200, 0, 0, 0.06)";
  return undefined;
}

function rowBackground(row: DraftBoardRow): string | undefined {
  return breakoutTint(row) ?? tierBackground(row.tier);
}

export interface DraftBoardTableProps {
  season: number;
  system?: string;
  version?: string;
}

export function DraftBoardTable({
  season,
  system = "zar",
  version = "1.0",
}: DraftBoardTableProps) {
  const { data, loading, error } = useQuery<BoardData, BoardVars>(BOARD_QUERY, {
    variables: { season, system, version },
  });

  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [positionFilter, setPositionFilter] = useState<string | null>(null);
  const [playerTypeFilter, setPlayerTypeFilter] = useState<PlayerTypeFilter>("all");
  const [searchQuery, setSearchQuery] = useState("");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "value" ? "desc" : "asc");
    }
  };

  const rows = useMemo(() => {
    if (!data) return [];
    let filtered = data.board.rows;

    if (playerTypeFilter !== "all") {
      filtered = filtered.filter((r) => r.playerType === playerTypeFilter);
    }
    if (positionFilter !== null) {
      filtered = filtered.filter((r) => r.position === positionFilter);
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter((r) => r.playerName.toLowerCase().includes(q));
    }

    return [...filtered].sort((a, b) => compareValues(a[sortKey], b[sortKey], sortDir));
  }, [data, sortKey, sortDir, positionFilter, playerTypeFilter, searchQuery]);

  if (loading) return <div className="p-4 text-gray-500">Loading board…</div>;
  if (error) return <div className="p-4 text-red-600">Error: {error.message}</div>;
  if (!data) return null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-3 items-center justify-between">
        <FilterBar
          playerTypeFilter={playerTypeFilter}
          onPlayerTypeChange={setPlayerTypeFilter}
          positionFilter={positionFilter}
          onPositionChange={setPositionFilter}
        />
        <SearchInput value={searchQuery} onChange={setSearchQuery} />
      </div>

      <div className="overflow-auto max-h-[calc(100vh-8rem)]">
        <table className="w-full text-sm border-collapse">
          <thead className="sticky top-0 z-10">
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left cursor-pointer select-none hover:bg-gray-200 whitespace-nowrap"
                >
                  {col.label}
                  {sortKey === col.key && (
                    <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={`${row.playerId}-${row.playerType}`}
                style={{ backgroundColor: rowBackground(row) }}
                className="hover:brightness-95"
              >
                <td className="border border-gray-200 px-2 py-1">{row.rank}</td>
                <td className="border border-gray-200 px-2 py-1 whitespace-nowrap">
                  {row.playerName}
                </td>
                <td className="border border-gray-200 px-2 py-1">{row.position}</td>
                <td className="border border-gray-200 px-2 py-1">{row.tier ?? ""}</td>
                <td className="border border-gray-200 px-2 py-1 font-mono">
                  ${row.value.toFixed(1)}
                </td>
                <td className="border border-gray-200 px-2 py-1">
                  {row.adpOverall != null ? row.adpOverall.toFixed(1) : ""}
                </td>
                <AdpDeltaCell delta={row.adpDelta} />
                <BreakoutCell rank={row.breakoutRank} type="breakout" />
                <BreakoutCell rank={row.bustRank} type="bust" />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AdpDeltaCell({ delta }: { delta: number | null }) {
  if (delta == null) {
    return <td className="border border-gray-200 px-2 py-1" />;
  }
  let className = "border border-gray-200 px-2 py-1";
  if (delta >= ADP_DELTA_THRESHOLD) className += " text-green-700 font-bold";
  else if (delta <= -ADP_DELTA_THRESHOLD) className += " text-red-700 font-bold";
  return <td className={className}>{delta}</td>;
}

function BreakoutCell({ rank, type }: { rank: number | null; type: "breakout" | "bust" }) {
  if (rank == null) {
    return <td className="border border-gray-200 px-2 py-1" />;
  }
  const label = type === "breakout" ? `B${rank}` : `X${rank}`;
  const colorClass = type === "breakout" ? "text-green-700" : "text-red-700";
  const title =
    type === "breakout"
      ? `#${rank} breakout candidate (within player type)`
      : `#${rank} bust risk (within player type)`;

  return (
    <td className={`border border-gray-200 px-2 py-1 ${colorClass}`} title={title}>
      {label}
    </td>
  );
}
