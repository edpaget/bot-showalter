import { useLazyQuery } from "@apollo/client";
import { useMemo, useState } from "react";
import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import { PROJECTIONS_QUERY } from "../graphql/queries";
import type { Projection } from "../types/analysis";

type SortKey = "playerName" | "system" | "version" | "playerType";
type SortDir = "asc" | "desc";

export function ProjectionsView({ season = 2026 }: { season?: number }) {
  const [searchInput, setSearchInput] = useState("");
  const [system, setSystem] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("playerName");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const { openPlayer } = usePlayerDrawer();

  const [fetchProjections, { data, loading, error }] = useLazyQuery<{
    projections: Projection[];
  }>(PROJECTIONS_QUERY);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchInput.trim()) return;
    fetchProjections({
      variables: {
        season,
        playerName: searchInput.trim(),
        system: system || undefined,
      },
    });
  };

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const projections = useMemo(() => {
    if (!data?.projections) return [];
    return [...data.projections].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const cmp = String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [data, sortKey, sortDir]);

  const columns: { key: SortKey; label: string }[] = [
    { key: "playerName", label: "Player" },
    { key: "system", label: "System" },
    { key: "version", label: "Version" },
    { key: "playerType", label: "Type" },
  ];

  return (
    <div className="p-4 flex flex-col gap-3">
      <h1 className="text-xl font-bold">Projections Lookup</h1>
      <form onSubmit={handleSubmit} className="flex gap-2 items-end">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Player Name</label>
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
            placeholder="Search player..."
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">System</label>
          <input
            type="text"
            value={system}
            onChange={(e) => setSystem(e.target.value)}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
            placeholder="All systems"
          />
        </div>
        <button type="submit" className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
          Search
        </button>
      </form>

      {loading && <p className="text-gray-500">Loading...</p>}
      {error && <p className="text-red-600">Error: {error.message}</p>}

      {projections.length > 0 && (
        <div className="overflow-auto">
          <table className="w-full text-sm border-collapse">
            <thead className="sticky top-0">
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
                <th className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left">Key Stats</th>
              </tr>
            </thead>
            <tbody>
              {projections.map((p, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="border border-gray-200 px-2 py-1">
                    <button
                      type="button"
                      onClick={() => openPlayer(0, p.playerName)}
                      className="text-blue-600 hover:underline"
                    >
                      {p.playerName}
                    </button>
                  </td>
                  <td className="border border-gray-200 px-2 py-1">{p.system}</td>
                  <td className="border border-gray-200 px-2 py-1">{p.version}</td>
                  <td className="border border-gray-200 px-2 py-1">{p.playerType}</td>
                  <td className="border border-gray-200 px-2 py-1 text-xs text-gray-600">
                    {Object.entries(p.stats)
                      .slice(0, 6)
                      .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(0) : v}`)
                      .join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
