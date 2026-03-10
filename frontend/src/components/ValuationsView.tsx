import { useQuery } from "@apollo/client";
import { useMemo, useState } from "react";
import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import type { ValuationsQuery } from "../generated/graphql";
import { VALUATIONS_QUERY } from "../graphql/queries";
import { displayPosition } from "../lib/position";

type SortKey = "rank" | "playerName" | "position" | "value" | "playerType" | "system" | "version";
type SortDir = "asc" | "desc";

export function ValuationsView({ season = 2026 }: { season?: number }) {
  const [playerType, setPlayerType] = useState("");
  const [position, setPosition] = useState("");
  const [topN, setTopN] = useState("100");
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const { openPlayer } = usePlayerDrawer();

  const { data, loading, error } = useQuery<ValuationsQuery>(VALUATIONS_QUERY, {
    variables: {
      season,
      playerType: playerType || undefined,
      position: position || undefined,
      top: topN ? parseInt(topN, 10) : undefined,
    },
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "value" ? "desc" : "asc");
    }
  };

  const rows = useMemo(() => {
    if (!data?.valuations) return [];
    return [...data.valuations].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      const cmp = String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [data, sortKey, sortDir]);

  const columns: { key: SortKey; label: string }[] = [
    { key: "rank", label: "Rank" },
    { key: "playerName", label: "Player" },
    { key: "position", label: "Pos" },
    { key: "value", label: "Value" },
    { key: "playerType", label: "Type" },
    { key: "system", label: "System" },
    { key: "version", label: "Version" },
  ];

  return (
    <div className="p-4 flex flex-col gap-3">
      <h1 className="text-xl font-bold">Valuation Rankings</h1>
      <div className="flex gap-3 items-end flex-wrap">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Player Type</label>
          <select
            value={playerType}
            onChange={(e) => setPlayerType(e.target.value)}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
          >
            <option value="">All</option>
            <option value="batter">Batter</option>
            <option value="pitcher">Pitcher</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Position</label>
          <input
            type="text"
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            className="border border-gray-300 rounded px-2 py-1 text-sm w-16"
            placeholder="All"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Top N</label>
          <input
            type="number"
            value={topN}
            onChange={(e) => setTopN(e.target.value)}
            className="border border-gray-300 rounded px-2 py-1 text-sm w-20"
          />
        </div>
      </div>

      {loading && <p className="text-gray-500">Loading...</p>}
      {error && <p className="text-red-600">Error: {error.message}</p>}

      {rows.length > 0 && (
        <div className="overflow-auto max-h-[calc(100vh-12rem)]">
          <table className="w-full text-sm border-collapse">
            <thead className="sticky top-0 z-10">
              <tr>
                {columns.map((col) => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left cursor-pointer select-none hover:bg-gray-200"
                  >
                    {col.label}
                    {sortKey === col.key && <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((v, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="border border-gray-200 px-2 py-1">{v.rank}</td>
                  <td className="border border-gray-200 px-2 py-1">
                    <button
                      type="button"
                      onClick={() => openPlayer(0, v.playerName)}
                      className="text-blue-600 hover:underline"
                    >
                      {v.playerName}
                    </button>
                  </td>
                  <td className="border border-gray-200 px-2 py-1">{displayPosition(v.position)}</td>
                  <td className="border border-gray-200 px-2 py-1 font-mono">${v.value.toFixed(1)}</td>
                  <td className="border border-gray-200 px-2 py-1">{v.playerType}</td>
                  <td className="border border-gray-200 px-2 py-1">{v.system}</td>
                  <td className="border border-gray-200 px-2 py-1">{v.version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
