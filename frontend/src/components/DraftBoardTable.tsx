import { useQuery } from "@apollo/client";
import { useMemo, useState } from "react";
import { ADP_DELTA_THRESHOLD, TIER_COLORS } from "../constants/tiers";
import type { BoardQuery, DraftBoardRowType } from "../generated/graphql";
import { BOARD_QUERY } from "../graphql/queries";
import { displayPosition } from "../lib/position";
import { FilterBar, type PlayerTypeFilter } from "./FilterBar";
import { SearchInput } from "./SearchInput";

type FixedSortKey = keyof Pick<
  DraftBoardRowType,
  "rank" | "playerName" | "position" | "tier" | "value" | "adpOverall" | "adpDelta" | "breakoutRank" | "bustRank"
>;

type SortKey = FixedSortKey | `z:${string}`;

type SortDir = "asc" | "desc";

const NAME_COLUMNS: { key: FixedSortKey; label: string }[] = [
  { key: "rank", label: "Rank" },
  { key: "playerName", label: "Player" },
];

const STAT_COLUMNS: { key: FixedSortKey; label: string }[] = [
  { key: "position", label: "Pos" },
  { key: "tier", label: "Tier" },
  { key: "value", label: "Value" },
];

const RIGHT_COLUMNS: { key: FixedSortKey; label: string }[] = [
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

function zScoreColor(z: number): string {
  if (z >= 1.5) return "text-green-700 font-semibold";
  if (z >= 0.5) return "text-green-600";
  if (z <= -1.5) return "text-red-700 font-semibold";
  if (z <= -0.5) return "text-red-600";
  return "text-gray-600";
}

const CATEGORY_LABELS: Record<string, string> = {
  hr: "HR",
  r: "R",
  rbi: "RBI",
  obp: "OBP",
  sb: "SB",
  era: "ERA",
  whip: "WHIP",
  so: "K",
  w: "W",
  "sv+hld": "SV+H",
};

function tierBackground(tier: number | null): string | undefined {
  if (tier == null) return undefined;
  return TIER_COLORS[(tier - 1) % TIER_COLORS.length]!;
}

function breakoutTint(row: DraftBoardRowType): string | undefined {
  if (row.breakoutRank != null && row.breakoutRank <= 20) return "rgba(0, 128, 0, 0.06)";
  if (row.bustRank != null && row.bustRank <= 20) return "rgba(200, 0, 0, 0.06)";
  return undefined;
}

function rowBackground(row: DraftBoardRowType): string | undefined {
  return breakoutTint(row) ?? tierBackground(row.tier);
}

export interface DraftBoardTableProps {
  season: number;
  system?: string;
  version?: string;
  draftedPlayerIds?: Set<number>;
  onDraft?: (playerId: number, position: string) => void;
  onPlayerClick?: (playerId: number, playerName: string, playerType: string) => void;
  sessionActive?: boolean;
  pickLoading?: boolean;
}

type StatusFilter = "all" | "available" | "drafted";

export function DraftBoardTable({
  season,
  system,
  version,
  draftedPlayerIds,
  onDraft,
  onPlayerClick,
  sessionActive = false,
  pickLoading,
}: DraftBoardTableProps) {
  const { data, loading, error } = useQuery<BoardQuery>(BOARD_QUERY, {
    variables: { season, system: system ?? null, version: version ?? null },
  });

  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [positionFilter, setPositionFilter] = useState<string | null>(null);
  const [playerTypeFilter, setPlayerTypeFilter] = useState<PlayerTypeFilter>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "value" || key.startsWith("z:") ? "desc" : "asc");
    }
  };

  const visibleCategories = useMemo(() => {
    if (!data) return { batting: [] as string[], pitching: [] as string[] };
    const { battingCategories, pitchingCategories } = data.board;
    if (playerTypeFilter === "pitcher") return { batting: [], pitching: pitchingCategories };
    if (playerTypeFilter === "batter") return { batting: battingCategories, pitching: [] };
    return { batting: battingCategories, pitching: pitchingCategories };
  }, [data, playerTypeFilter]);

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
    if (draftedPlayerIds && statusFilter !== "all") {
      if (statusFilter === "available") {
        filtered = filtered.filter((r) => !draftedPlayerIds.has(r.playerId));
      } else {
        filtered = filtered.filter((r) => draftedPlayerIds.has(r.playerId));
      }
    }

    return [...filtered].sort((a, b) => {
      if (sortKey.startsWith("z:")) {
        const cat = sortKey.slice(2);
        const aVal = (a.categoryZScores[cat] as number | undefined) ?? null;
        const bVal = (b.categoryZScores[cat] as number | undefined) ?? null;
        return compareValues(aVal, bVal, sortDir);
      }
      return compareValues(a[sortKey as FixedSortKey], b[sortKey as FixedSortKey], sortDir);
    });
  }, [data, sortKey, sortDir, positionFilter, playerTypeFilter, searchQuery, draftedPlayerIds, statusFilter]);

  if (loading) return <div className="p-4 text-gray-500">Loading board…</div>;
  if (error) return <div className="p-4 text-red-600">Error: {error.message}</div>;
  if (!data) return null;

  return (
    <div className="flex flex-col gap-3 h-full">
      <div className="flex flex-wrap gap-3 items-center justify-between">
        <FilterBar
          playerTypeFilter={playerTypeFilter}
          onPlayerTypeChange={setPlayerTypeFilter}
          positionFilter={positionFilter}
          onPositionChange={setPositionFilter}
        />
        <div className="flex gap-2 items-center">
          {draftedPlayerIds && (
            <div className="flex gap-1">
              {(["all", "available", "drafted"] as StatusFilter[]).map((s) => (
                <button
                  type="button"
                  key={s}
                  onClick={() => setStatusFilter(s)}
                  className={`px-2 py-1 text-xs rounded ${
                    statusFilter === s ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {s === "all" ? "All" : s === "available" ? "Available" : "Drafted"}
                </button>
              ))}
            </div>
          )}
          <SearchInput value={searchQuery} onChange={setSearchQuery} />
        </div>
      </div>

      <div className="overflow-auto flex-1 min-h-0">
        <table className="w-full text-sm border-collapse">
          <thead className="sticky top-0 z-10">
            <tr>
              {NAME_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left cursor-pointer select-none hover:bg-gray-200 whitespace-nowrap"
                >
                  {col.label}
                  {sortKey === col.key && <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>}
                </th>
              ))}
              {sessionActive && (
                <th className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left whitespace-nowrap">Action</th>
              )}
              {STAT_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left cursor-pointer select-none hover:bg-gray-200 whitespace-nowrap"
                >
                  {col.label}
                  {sortKey === col.key && <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>}
                </th>
              ))}
              {visibleCategories.batting.map((cat) => (
                <th
                  key={`bat-${cat}`}
                  onClick={() => handleSort(`z:${cat}`)}
                  className="bg-green-50 border border-gray-300 px-2 py-1.5 text-right cursor-pointer select-none hover:bg-green-100 whitespace-nowrap"
                >
                  {CATEGORY_LABELS[cat] ?? cat.toUpperCase()}
                  {sortKey === `z:${cat}` && <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>}
                </th>
              ))}
              {visibleCategories.pitching.map((cat) => (
                <th
                  key={`pit-${cat}`}
                  onClick={() => handleSort(`z:${cat}`)}
                  className="bg-blue-50 border border-gray-300 px-2 py-1.5 text-right cursor-pointer select-none hover:bg-blue-100 whitespace-nowrap"
                >
                  {CATEGORY_LABELS[cat] ?? cat.toUpperCase()}
                  {sortKey === `z:${cat}` && <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>}
                </th>
              ))}
              {RIGHT_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left cursor-pointer select-none hover:bg-gray-200 whitespace-nowrap"
                >
                  {col.label}
                  {sortKey === col.key && <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isDrafted = draftedPlayerIds?.has(row.playerId) ?? false;
              return (
                <tr
                  key={`${row.playerId}-${row.playerType}`}
                  style={{ backgroundColor: rowBackground(row) }}
                  className={`hover:brightness-95 ${isDrafted ? "opacity-40" : ""}`}
                >
                  <td className="border border-gray-200 px-2 py-1">{row.rank}</td>
                  <td className="border border-gray-200 px-2 py-1 whitespace-nowrap">
                    {onPlayerClick ? (
                      <button
                        type="button"
                        onClick={() => onPlayerClick(row.playerId, row.playerName, row.playerType)}
                        className="text-blue-600 hover:underline"
                      >
                        {row.playerName}
                      </button>
                    ) : (
                      row.playerName
                    )}
                  </td>
                  {sessionActive && (
                    <td className="border border-gray-200 px-2 py-1">
                      {!isDrafted && onDraft && (
                        <button
                          type="button"
                          disabled={pickLoading}
                          onClick={() => onDraft(row.playerId, row.position)}
                          className="px-2 py-0.5 text-xs bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          Draft
                        </button>
                      )}
                    </td>
                  )}
                  <td className="border border-gray-200 px-2 py-1">{displayPosition(row.position)}</td>
                  <td className="border border-gray-200 px-2 py-1">{row.tier ?? ""}</td>
                  <td className="border border-gray-200 px-2 py-1 font-mono">${row.value.toFixed(1)}</td>
                  {visibleCategories.batting.map((cat) => {
                    const z =
                      row.playerType !== "pitcher" ? (row.categoryZScores[cat] as number | undefined) : undefined;
                    return (
                      <td
                        key={`bat-${cat}`}
                        className={`border border-gray-200 px-2 py-1 text-right font-mono text-xs ${z != null ? zScoreColor(z) : ""}`}
                      >
                        {z != null ? z.toFixed(1) : ""}
                      </td>
                    );
                  })}
                  {visibleCategories.pitching.map((cat) => {
                    const z =
                      row.playerType === "pitcher" ? (row.categoryZScores[cat] as number | undefined) : undefined;
                    return (
                      <td
                        key={`pit-${cat}`}
                        className={`border border-gray-200 px-2 py-1 text-right font-mono text-xs ${z != null ? zScoreColor(z) : ""}`}
                      >
                        {z != null ? z.toFixed(1) : ""}
                      </td>
                    );
                  })}
                  <td className="border border-gray-200 px-2 py-1">
                    {row.adpOverall != null ? row.adpOverall.toFixed(1) : ""}
                  </td>
                  <AdpDeltaCell delta={row.adpDelta} />
                  <BreakoutCell rank={row.breakoutRank} type="breakout" />
                  <BreakoutCell rank={row.bustRank} type="bust" />
                </tr>
              );
            })}
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
