import { useQuery } from "@apollo/client";
import { useMemo, useState } from "react";
import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import type { CategoryConfigType, LeagueQuery, ProjectionBoardQuery, WebConfigQuery } from "../generated/graphql";
import { LEAGUE_QUERY, PROJECTION_BOARD_QUERY, WEB_CONFIG_QUERY } from "../graphql/queries";

type SortKey = "playerName" | "playerType" | string;
type SortDir = "asc" | "desc";

function formatStat(value: number | undefined, statType: string): string {
  if (value == null) return "—";
  if (statType === "rate") return value.toFixed(3);
  return Math.round(value).toString();
}

export function ProjectionsView({ season = 2026 }: { season?: number }) {
  const [nameFilter, setNameFilter] = useState("");
  const [playerTypeFilter, setPlayerTypeFilter] = useState<"batter" | "pitcher">("batter");
  const [selectedSystemIdx, setSelectedSystemIdx] = useState(0);
  const [sortKey, setSortKey] = useState<SortKey>("playerName");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const { openPlayer } = usePlayerDrawer();

  const { data: configData } = useQuery<WebConfigQuery>(WEB_CONFIG_QUERY);
  const { data: leagueData } = useQuery<LeagueQuery>(LEAGUE_QUERY);

  const systems = configData?.webConfig.projections ?? [];
  const selected = systems[selectedSystemIdx];

  const { data, loading, error } = useQuery<ProjectionBoardQuery>(PROJECTION_BOARD_QUERY, {
    variables: {
      season,
      system: selected?.system ?? "",
      version: selected?.version ?? "",
      playerType: playerTypeFilter,
    },
    skip: !selected,
  });

  const categories: CategoryConfigType[] =
    playerTypeFilter === "pitcher"
      ? (leagueData?.league.pitchingCategories ?? [])
      : (leagueData?.league.battingCategories ?? []);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const projections = useMemo(() => {
    if (!data?.projectionBoard) return [];
    let rows = [...data.projectionBoard];

    if (nameFilter.trim()) {
      const q = nameFilter.trim().toLowerCase();
      rows = rows.filter((p) => p.playerName.toLowerCase().includes(q));
    }

    return rows.sort((a, b) => {
      let cmp: number;
      if (sortKey === "playerName" || sortKey === "playerType") {
        cmp = String(a[sortKey]).localeCompare(String(b[sortKey]));
      } else {
        const av = (a.stats[sortKey] as number) ?? -Infinity;
        const bv = (b.stats[sortKey] as number) ?? -Infinity;
        cmp = av - bv;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [data, nameFilter, sortKey, sortDir]);

  return (
    <div className="p-4 flex flex-col gap-3">
      <h1 className="text-xl font-bold">Projections</h1>

      <div className="flex gap-3 items-end flex-wrap">
        {systems.length > 1 && (
          <div>
            <label className="block text-xs text-gray-500 mb-1">System</label>
            <select
              value={selectedSystemIdx}
              onChange={(e) => setSelectedSystemIdx(Number(e.target.value))}
              className="border border-gray-300 rounded px-2 py-1 text-sm"
            >
              {systems.map((sv, i) => (
                <option key={`${sv.system}-${sv.version}`} value={i}>
                  {sv.system} {sv.version}
                </option>
              ))}
            </select>
          </div>
        )}
        <div>
          <label className="block text-xs text-gray-500 mb-1">Type</label>
          <select
            value={playerTypeFilter}
            onChange={(e) => setPlayerTypeFilter(e.target.value as "batter" | "pitcher")}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
          >
            <option value="batter">Batters</option>
            <option value="pitcher">Pitchers</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Search</label>
          <input
            type="text"
            value={nameFilter}
            onChange={(e) => setNameFilter(e.target.value)}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
            placeholder="Filter by name..."
          />
        </div>
        {selected && <span className="text-xs text-gray-400 self-center">{projections.length} players</span>}
      </div>

      {loading && <p className="text-gray-500">Loading...</p>}
      {error && <p className="text-red-600">Error: {error.message}</p>}

      {projections.length > 0 && (
        <div className="overflow-auto max-h-[calc(100vh-180px)]">
          <table className="w-full text-sm border-collapse">
            <thead className="sticky top-0 z-10">
              <tr>
                <th
                  onClick={() => handleSort("playerName")}
                  className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left cursor-pointer select-none hover:bg-gray-200"
                >
                  Player
                  {sortKey === "playerName" && <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>}
                </th>
                {categories.map((cat) => (
                  <th
                    key={cat.key}
                    onClick={() => handleSort(cat.key)}
                    className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-right cursor-pointer select-none hover:bg-gray-200 whitespace-nowrap"
                  >
                    {cat.name}
                    {sortKey === cat.key && <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {projections.map((p) => (
                <tr key={p.playerId ?? p.playerName} className="hover:bg-gray-50">
                  <td className="border border-gray-200 px-2 py-1">
                    <button
                      type="button"
                      onClick={() => openPlayer(p.playerId ?? 0, p.playerName, p.playerType)}
                      className="text-blue-600 hover:underline"
                    >
                      {p.playerName}
                    </button>
                  </td>
                  {categories.map((cat) => (
                    <td key={cat.key} className="border border-gray-200 px-2 py-1 text-right font-mono text-xs">
                      {formatStat(p.stats[cat.key] as number | undefined, cat.statType)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
