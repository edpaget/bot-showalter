import { useState } from "react";
import { useLazyQuery } from "@apollo/client";
import { ARBITRAGE_QUERY } from "../graphql/queries";
import type { ArbitrageReport, FallingPlayer } from "../types/session";

const POSITION_FILTERS = ["All", "C", "1B", "2B", "SS", "3B", "OF", "SP", "RP"] as const;

interface ArbitragePanelProps {
  arbitrage: ArbitrageReport | null;
  sessionId: number;
  onDraft: (playerId: number, position: string) => void;
}

export function ArbitragePanel({ arbitrage, sessionId, onDraft }: ArbitragePanelProps) {
  const [tab, setTab] = useState<"falling" | "reaches">("falling");
  const [posFilter, setPosFilter] = useState<string>("All");
  const [threshold, setThreshold] = useState<number>(10);
  const [localReport, setLocalReport] = useState<ArbitrageReport | null>(null);

  const [fetchArbitrage] = useLazyQuery<{ arbitrage: ArbitrageReport }>(ARBITRAGE_QUERY, {
    fetchPolicy: "network-only",
    onCompleted: (data) => setLocalReport(data.arbitrage),
  });

  const report = localReport ?? arbitrage;

  const handleThresholdChange = (value: number) => {
    setThreshold(value);
    fetchArbitrage({
      variables: {
        sessionId,
        threshold: value,
        position: posFilter === "All" ? null : posFilter,
      },
    });
  };

  const falling = report?.falling ?? [];
  const reaches = report?.reaches ?? [];

  const filteredFalling =
    posFilter === "All" ? falling : falling.filter((f) => f.position === posFilter);

  return (
    <div className="border border-gray-200 rounded p-3">
      <h3 className="text-sm font-semibold mb-2">ADP Arbitrage</h3>

      <div className="flex gap-1 mb-2">
        <button
          onClick={() => setTab("falling")}
          className={`px-2 py-0.5 text-xs rounded ${
            tab === "falling" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          Falling ({falling.length})
        </button>
        <button
          onClick={() => setTab("reaches")}
          className={`px-2 py-0.5 text-xs rounded ${
            tab === "reaches" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          Reaches ({reaches.length})
        </button>
      </div>

      {tab === "falling" && (
        <>
          <div className="flex items-center gap-2 mb-2">
            <div className="flex flex-wrap gap-1">
              {POSITION_FILTERS.map((pos) => (
                <button
                  key={pos}
                  onClick={() => setPosFilter(pos)}
                  className={`px-2 py-0.5 text-xs rounded ${
                    posFilter === pos
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {pos}
                </button>
              ))}
            </div>
            <label className="text-xs text-gray-500 flex items-center gap-1 ml-auto">
              Slip
              <input
                type="number"
                min={1}
                max={100}
                value={threshold}
                onChange={(e) => handleThresholdChange(Number(e.target.value))}
                className="w-12 px-1 py-0.5 text-xs border rounded"
              />
            </label>
          </div>
          <FallingTable players={filteredFalling} onDraft={onDraft} />
        </>
      )}

      {tab === "reaches" && (
        <ReachesTable reaches={reaches} />
      )}
    </div>
  );
}

function FallingTable({
  players,
  onDraft,
}: {
  players: FallingPlayer[];
  onDraft: (playerId: number, position: string) => void;
}) {
  if (players.length === 0) {
    return <p className="text-xs text-gray-500">No falling players detected</p>;
  }
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-gray-500">
          <th className="pb-1">Player</th>
          <th className="pb-1">Pos</th>
          <th className="pb-1">ADP</th>
          <th className="pb-1">Slip</th>
          <th className="pb-1">Value</th>
          <th className="pb-1">Score</th>
          <th className="pb-1" />
        </tr>
      </thead>
      <tbody>
        {players.map((fp) => (
          <tr key={fp.playerId} className="border-t border-gray-100">
            <td className="py-1">{fp.playerName}</td>
            <td className="py-1">{fp.position}</td>
            <td className="py-1 font-mono">{fp.adp.toFixed(0)}</td>
            <td className="py-1 font-mono text-red-600">+{fp.picksPastAdp.toFixed(0)}</td>
            <td className="py-1 font-mono">${fp.value.toFixed(1)}</td>
            <td className="py-1 font-mono">{fp.arbitrageScore.toFixed(1)}</td>
            <td className="py-1">
              <button
                onClick={() => onDraft(fp.playerId, fp.position.toUpperCase())}
                className="px-2 py-0.5 text-xs bg-green-600 text-white rounded hover:bg-green-700"
              >
                Draft
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ReachesTable({ reaches }: { reaches: ArbitrageReport["reaches"] }) {
  if (reaches.length === 0) {
    return <p className="text-xs text-gray-500">No reach picks detected</p>;
  }
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-gray-500">
          <th className="pb-1">Player</th>
          <th className="pb-1">Pos</th>
          <th className="pb-1">ADP</th>
          <th className="pb-1">Pick#</th>
          <th className="pb-1">Ahead</th>
          <th className="pb-1">Team</th>
        </tr>
      </thead>
      <tbody>
        {reaches.map((rp) => (
          <tr key={`${rp.playerId}-${rp.pickNumber}`} className="border-t border-gray-100">
            <td className="py-1">{rp.playerName}</td>
            <td className="py-1">{rp.position}</td>
            <td className="py-1 font-mono">{rp.adp.toFixed(0)}</td>
            <td className="py-1 font-mono">{rp.pickNumber}</td>
            <td className="py-1 font-mono text-orange-600">-{rp.picksAheadOfAdp.toFixed(0)}</td>
            <td className="py-1">T{rp.drafterTeam}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
