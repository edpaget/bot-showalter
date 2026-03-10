import { useState } from "react";
import type { Recommendation } from "../types/session";
import { Position, displayPosition } from "../types/position";

const POSITION_FILTERS = ["All", ...Object.values(Position)] as const;

interface RecommendationPanelProps {
  recommendations: Recommendation[];
  onDraft: (playerId: number, position: string) => void;
  onPlayerClick?: (playerId: number, playerName: string) => void;
  sessionActive: boolean;
}

export function RecommendationPanel({
  recommendations,
  onDraft,
  onPlayerClick,
  sessionActive,
}: RecommendationPanelProps) {
  const [posFilter, setPosFilter] = useState<string>("All");

  const filtered =
    posFilter === "All"
      ? recommendations
      : recommendations.filter((r) => r.position === posFilter);

  return (
    <div className="border border-gray-200 rounded p-3">
      <h3 className="text-sm font-semibold mb-2">Recommendations</h3>
      <div className="flex flex-wrap gap-1 mb-2">
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
            {pos === "All" ? "All" : displayPosition(pos)}
          </button>
        ))}
      </div>
      {filtered.length === 0 ? (
        <p className="text-xs text-gray-500">No recommendations</p>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-gray-500">
              <th className="pb-1">Player</th>
              <th className="pb-1">Pos</th>
              <th className="pb-1">Value</th>
              <th className="pb-1">Score</th>
              <th className="pb-1" />
            </tr>
          </thead>
          <tbody>
            {filtered.map((rec) => (
              <tr key={rec.playerId} className="border-t border-gray-100">
                <td className="py-1" title={rec.reason}>
                  {onPlayerClick ? (
                    <button
                      onClick={() => onPlayerClick(rec.playerId, rec.playerName)}
                      className="text-blue-600 hover:underline"
                    >
                      {rec.playerName}
                    </button>
                  ) : (
                    rec.playerName
                  )}
                </td>
                <td className="py-1">{displayPosition(rec.position)}</td>
                <td className="py-1 font-mono">${rec.value.toFixed(1)}</td>
                <td className="py-1 font-mono">{rec.score.toFixed(2)}</td>
                <td className="py-1">
                  {sessionActive && (
                    <button
                      onClick={() => onDraft(rec.playerId, rec.position)}
                      className="px-2 py-0.5 text-xs bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      Draft
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
