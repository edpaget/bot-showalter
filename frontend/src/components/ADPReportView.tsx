import { useState, useMemo } from "react";
import { useQuery } from "@apollo/client";
import { ADP_REPORT_QUERY } from "../graphql/queries";
import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import { displayPosition } from "../types/position";
import type { ADPReport, ADPReportRow } from "../types/analysis";

type SortKey = "playerName" | "zarRank" | "zarValue" | "adpRank" | "rankDelta";
type SortDir = "asc" | "desc";

function compareValues(a: string | number, b: string | number, dir: SortDir): number {
  const cmp = typeof a === "string" ? a.localeCompare(b as string) : (a as number) - (b as number);
  return dir === "asc" ? cmp : -cmp;
}

function ADPSection({
  title,
  rows,
  openPlayer,
}: {
  title: string;
  rows: ADPReportRow[];
  openPlayer: (id: number, name: string) => void;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("rankDelta");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "zarValue" || key === "rankDelta" ? "desc" : "asc");
    }
  };

  const sorted = useMemo(
    () => [...rows].sort((a, b) => compareValues(a[sortKey], b[sortKey], sortDir)),
    [rows, sortKey, sortDir],
  );

  const columns: { key: SortKey; label: string }[] = [
    { key: "playerName", label: "Player" },
    { key: "zarRank", label: "ZAR Rank" },
    { key: "zarValue", label: "Value" },
    { key: "adpRank", label: "ADP Rank" },
    { key: "rankDelta", label: "Delta" },
  ];

  if (rows.length === 0) {
    return (
      <div className="mb-4">
        <h2 className="font-semibold mb-1">{title}</h2>
        <p className="text-gray-400 text-sm">None</p>
      </div>
    );
  }

  return (
    <div className="mb-4">
      <h2 className="font-semibold mb-1">
        {title} ({rows.length})
      </h2>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left cursor-pointer select-none hover:bg-gray-200"
              >
                {col.label}
                {sortKey === col.key && (
                  <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>
                )}
              </th>
            ))}
            <th className="bg-gray-100 border border-gray-300 px-2 py-1.5 text-left">Pos</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const deltaColor =
              r.rankDelta > 0
                ? "text-green-700 font-bold"
                : r.rankDelta < 0
                  ? "text-red-700 font-bold"
                  : "";
            return (
              <tr key={r.playerId} className="hover:bg-gray-50">
                <td className="border border-gray-200 px-2 py-1">
                  <button
                    onClick={() => openPlayer(r.playerId, r.playerName)}
                    className="text-blue-600 hover:underline"
                  >
                    {r.playerName}
                  </button>
                </td>
                <td className="border border-gray-200 px-2 py-1">{r.zarRank}</td>
                <td className="border border-gray-200 px-2 py-1 font-mono">
                  ${r.zarValue.toFixed(1)}
                </td>
                <td className="border border-gray-200 px-2 py-1">
                  {r.adpRank > 0 ? r.adpRank : "—"}
                </td>
                <td className={`border border-gray-200 px-2 py-1 ${deltaColor}`}>
                  {r.rankDelta > 0 ? `+${r.rankDelta}` : r.rankDelta}
                </td>
                <td className="border border-gray-200 px-2 py-1">
                  {displayPosition(r.position)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ADPReportView({ season = 2026 }: { season?: number }) {
  const { openPlayer } = usePlayerDrawer();
  const { data, loading, error } = useQuery<{ adpReport: ADPReport }>(ADP_REPORT_QUERY, {
    variables: { season },
  });

  if (loading) return <div className="p-4 text-gray-500">Loading ADP report...</div>;
  if (error) return <div className="p-4 text-red-600">Error: {error.message}</div>;
  if (!data) return null;

  const report = data.adpReport;

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-1">ADP Report</h1>
      <p className="text-sm text-gray-500 mb-4">
        {report.system} v{report.version} vs {report.provider} ADP · {report.nMatched} matched
      </p>
      <ADPSection title="Buy Targets" rows={report.buyTargets} openPlayer={openPlayer} />
      <ADPSection title="Avoid List" rows={report.avoidList} openPlayer={openPlayer} />
      <ADPSection
        title="Unranked Valuable"
        rows={report.unrankedValuable}
        openPlayer={openPlayer}
      />
    </div>
  );
}
