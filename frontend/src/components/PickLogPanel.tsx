import { useState } from "react";
import type { DraftPickType, DraftTradeType } from "../generated/graphql";
import { displayPosition } from "../lib/position";
import { snakeTeam } from "../lib/snakePicks";
import { Spinner } from "./Spinner";

interface PickLogPanelProps {
  picks: DraftPickType[];
  trades?: DraftTradeType[];
  teams?: number;
  userTeam?: number;
  teamNames?: Record<number, string>;
  onPlayerClick?: (playerId: number, playerName: string, playerType?: string) => void;
  onUndoTrade?: () => void;
  undoingTrade?: boolean;
}

function teamLabel(id: number, teamNames?: Record<number, string>): string {
  return teamNames?.[id] ?? `Team ${id}`;
}

export function PickLogPanel({
  picks,
  trades = [],
  teams = 0,
  userTeam = 0,
  teamNames,
  onPlayerClick,
  onUndoTrade,
  undoingTrade,
}: PickLogPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const latestPickNumber = picks.length > 0 ? picks[picks.length - 1]?.pickNumber : null;

  return (
    <div className="border border-gray-200 rounded bg-white flex-shrink-0">
      <button
        type="button"
        onClick={() => setCollapsed(!collapsed)}
        className="w-full px-3 py-2 text-sm font-semibold text-left flex justify-between items-center hover:bg-gray-50"
      >
        <span>Pick Log ({picks.length} picks)</span>
        <span>{collapsed ? "▶" : "▼"}</span>
      </button>
      {!collapsed && trades.length > 0 && (
        <div className="px-3 py-2 bg-indigo-50 border-b border-indigo-100">
          <p className="text-xs font-medium text-indigo-700 mb-1">Trades</p>
          {trades.map((trade, i) => (
            <div
              key={`trade-${trade.teamA}-${trade.teamB}-${i}`}
              className="flex items-center gap-2 text-xs text-indigo-600"
            >
              <span>
                {teamLabel(trade.teamA, teamNames)} ↔ {teamLabel(trade.teamB, teamNames)}: gave picks [
                {trade.teamAGives.join(", ")}] for picks [{trade.teamBGives.join(", ")}]
              </span>
              {i === trades.length - 1 && (trade.teamA === userTeam || trade.teamB === userTeam) && onUndoTrade && (
                <button
                  type="button"
                  onClick={onUndoTrade}
                  disabled={undoingTrade}
                  className="px-1.5 py-0.5 text-[10px] bg-indigo-200 text-indigo-700 rounded hover:bg-indigo-300 disabled:opacity-50 flex items-center gap-1"
                >
                  {undoingTrade && <Spinner className="h-2 w-2" />}
                  Undo
                </button>
              )}
            </div>
          ))}
        </div>
      )}
      {!collapsed && (
        <div className="max-h-40 overflow-auto">
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
                    <td className="py-1">
                      {teamLabel(pick.team, teamNames)}
                      {teams > 0 && snakeTeam(pick.pickNumber, teams) !== pick.team && (
                        <span className="ml-1 bg-indigo-100 text-indigo-700 text-[10px] px-1 rounded">traded</span>
                      )}
                    </td>
                    <td className="py-1">
                      {onPlayerClick ? (
                        <button
                          type="button"
                          onClick={() =>
                            onPlayerClick(
                              pick.playerId,
                              pick.playerName,
                              ["SP", "RP", "P"].includes(pick.position) ? "pitcher" : "batter",
                            )
                          }
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
