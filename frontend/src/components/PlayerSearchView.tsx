import { useLazyQuery } from "@apollo/client";
import { useMemo, useState } from "react";
import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import type { PlayerSearchQuery } from "../generated/graphql";
import { PLAYER_SEARCH_QUERY } from "../graphql/queries";

type SortKey = "name" | "team" | "age" | "primaryPosition" | "experience";
type SortDir = "asc" | "desc";

export function PlayerSearchView({ season = 2026 }: { season?: number }) {
  const [searchInput, setSearchInput] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const { openPlayer } = usePlayerDrawer();

  const [fetchPlayers, { data, loading, error }] = useLazyQuery<PlayerSearchQuery>(PLAYER_SEARCH_QUERY);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchInput.trim()) return;
    fetchPlayers({ variables: { name: searchInput.trim(), season } });
  };

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const players = useMemo(() => {
    if (!data?.playerSearch) return [];
    return [...data.playerSearch].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      const cmp = typeof av === "number" ? (av as number) - (bv as number) : String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [data, sortKey, sortDir]);

  const columns: { key: SortKey; label: string }[] = [
    { key: "name", label: "Name" },
    { key: "team", label: "Team" },
    { key: "age", label: "Age" },
    { key: "primaryPosition", label: "Pos" },
    { key: "experience", label: "Exp" },
  ];

  return (
    <div className="p-4 flex flex-col gap-3">
      <h1 className="text-xl font-bold">Player Search</h1>
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
        <button type="submit" className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
          Search
        </button>
      </form>

      {loading && <p className="text-gray-500">Loading...</p>}
      {error && <p className="text-red-600">Error: {error.message}</p>}

      {players.length > 0 && (
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
                <th className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left">Bats/Throws</th>
              </tr>
            </thead>
            <tbody>
              {players.map((p) => (
                <tr key={p.playerId} className="hover:bg-gray-50">
                  <td className="border border-gray-200 px-2 py-1">
                    <button
                      type="button"
                      onClick={() =>
                        openPlayer(
                          p.playerId,
                          p.name,
                          ["SP", "RP", "P"].includes(p.primaryPosition) ? "pitcher" : "batter",
                        )
                      }
                      className="text-blue-600 hover:underline"
                    >
                      {p.name}
                    </button>
                  </td>
                  <td className="border border-gray-200 px-2 py-1">{p.team}</td>
                  <td className="border border-gray-200 px-2 py-1">{p.age ?? "—"}</td>
                  <td className="border border-gray-200 px-2 py-1">{p.primaryPosition}</td>
                  <td className="border border-gray-200 px-2 py-1">{p.experience}</td>
                  <td className="border border-gray-200 px-2 py-1">
                    {p.bats ?? "—"}/{p.throws ?? "—"}
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
