import { useState } from "react";
import { displayPosition } from "../types/position";
import type { DraftPick } from "../types/session";

interface PickLogPanelProps {
  picks: DraftPick[];
  onPlayerClick?: (playerId: number, playerName: string) => void;
}

export function PickLogPanel({ picks, onPlayerClick }: PickLogPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const latestPickNumber = picks.length > 0 ? picks[picks.length - 1]?.pickNumber : null;

  return (
    <div className="border border-gray-200 rounded">
      <button
        type="button"
        onClick={() => setCollapsed(!collapsed)}
        className="w-full px-3 py-2 text-sm font-semibold text-left flex justify-between items-center hover:bg-gray-50"
      >
        <span>Pick Log ({picks.length} picks)</span>
        <span>{collapsed ? "▶" : "▼"}</span>
      </button>
      {!collapsed && (
        <div className="max-h-48 overflow-auto">
          {picks.length === 0 ? (
            <p className="px-3 py-2 text-xs text-gray-500">No picks yet</p>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-gray-500 sticky top-0 bg-white">
                  <th className="px-3 pb-1">Pick</th>
                  <th className="pb-1">Team</th>
                  <th className="pb-1">Player</th>
                  <th className="pb-1">Pos</th>
                  <th className="pb-1">Price</th>
                </tr>
              </thead>
              <tbody>
                {[...picks].reverse().map((pick) => (
                  <tr
                    key={pick.pickNumber}
                    className={`border-t border-gray-100 ${
                      pick.pickNumber === latestPickNumber ? "bg-yellow-50 font-medium" : ""
                    }`}
                  >
                    <td className="px-3 py-1">{pick.pickNumber}</td>
                    <td className="py-1">Team {pick.team}</td>
                    <td className="py-1">
                      {onPlayerClick ? (
                        <button
                          type="button"
                          onClick={() => onPlayerClick(pick.playerId, pick.playerName)}
                          className="text-blue-600 hover:underline"
                        >
                          {pick.playerName}
                        </button>
                      ) : (
                        pick.playerName
                      )}
                    </td>
                    <td className="py-1">{displayPosition(pick.position)}</td>
                    <td className="py-1">{pick.price != null ? `$${pick.price}` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
