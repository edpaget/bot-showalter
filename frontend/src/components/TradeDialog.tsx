import { useLazyQuery, useMutation } from "@apollo/client";
import { useCallback, useState } from "react";
import type { DraftStateType, DraftTradeType, EvaluateTradeQuery, TradePicksMutation } from "../generated/graphql";
import { TRADE_PICKS } from "../graphql/mutations";
import { EVALUATE_TRADE_QUERY } from "../graphql/queries";
import { remainingPicksForTeam } from "../lib/snakePicks";
import { Spinner } from "./Spinner";

interface TradeDialogProps {
  sessionId: number;
  userTeam: number;
  teams: number;
  currentPick: number;
  totalPicks: number;
  trades: DraftTradeType[];
  onTradeComplete: (state: DraftStateType) => void;
  onClose: () => void;
}

export function TradeDialog({
  sessionId,
  userTeam,
  teams,
  currentPick,
  totalPicks,
  trades,
  onTradeComplete,
  onClose,
}: TradeDialogProps) {
  const [teamA, setTeamA] = useState<number>(userTeam);
  const [teamB, setTeamB] = useState<number>(userTeam === 1 ? 2 : 1);
  const [selectedGives, setSelectedGives] = useState<Set<number>>(new Set());
  const [selectedReceives, setSelectedReceives] = useState<Set<number>>(new Set());

  const [evaluate, { data: evalData, loading: evaluating }] = useLazyQuery<EvaluateTradeQuery>(EVALUATE_TRADE_QUERY);
  const [executeTrade, { loading: executing }] = useMutation<TradePicksMutation>(TRADE_PICKS);

  const teamAPicks = remainingPicksForTeam(teamA, currentPick, totalPicks, teams, trades);
  const teamBPicks = remainingPicksForTeam(teamB, currentPick, totalPicks, teams, trades);

  const allTeams = Array.from({ length: teams }, (_, i) => i + 1);

  const handleTeamAChange = useCallback((newTeam: number) => {
    setTeamA(newTeam);
    setSelectedGives(new Set());
  }, []);

  const handleTeamBChange = useCallback((newTeam: number) => {
    setTeamB(newTeam);
    setSelectedReceives(new Set());
  }, []);

  const toggleGive = useCallback((pick: number) => {
    setSelectedGives((prev) => {
      const next = new Set(prev);
      if (next.has(pick)) next.delete(pick);
      else next.add(pick);
      return next;
    });
  }, []);

  const toggleReceive = useCallback((pick: number) => {
    setSelectedReceives((prev) => {
      const next = new Set(prev);
      if (next.has(pick)) next.delete(pick);
      else next.add(pick);
      return next;
    });
  }, []);

  const hasSelection = selectedGives.size > 0 && selectedReceives.size > 0;

  const handleEvaluate = useCallback(async () => {
    if (!hasSelection) return;
    await evaluate({
      variables: {
        sessionId,
        gives: [...selectedGives],
        receives: [...selectedReceives],
      },
    });
  }, [sessionId, selectedGives, selectedReceives, hasSelection, evaluate]);

  const handleExecute = useCallback(async () => {
    if (!hasSelection) return;
    const result = await executeTrade({
      variables: {
        sessionId,
        gives: [...selectedGives],
        receives: [...selectedReceives],
        partnerTeam: teamB,
        teamA,
      },
    });
    if (result.data) {
      onTradeComplete(result.data.tradePicks);
      onClose();
    }
  }, [sessionId, selectedGives, selectedReceives, teamA, teamB, hasSelection, executeTrade, onTradeComplete, onClose]);

  const evaluation = evalData?.evaluateTrade;

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center">
      <button type="button" aria-label="Close" className="fixed inset-0 bg-black/30 cursor-default" onClick={onClose} />
      <div className="relative z-50 bg-white rounded-lg shadow-xl max-w-xl w-full mx-4 p-5">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Trade Picks</h2>
          <button type="button" onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl leading-none">
            ×
          </button>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Team A</label>
            <select
              value={teamA}
              onChange={(e) => handleTeamAChange(Number(e.target.value))}
              className="border border-gray-300 rounded px-2 py-1 text-sm w-full"
            >
              {allTeams
                .filter((t) => t !== teamB)
                .map((t) => (
                  <option key={t} value={t}>
                    Team {t}
                  </option>
                ))}
            </select>
            <h3 className="text-sm font-medium mt-2 mb-1">Gives</h3>
            <div className="space-y-1 max-h-48 overflow-auto">
              {teamAPicks.map((pick) => (
                <label key={pick} className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" checked={selectedGives.has(pick)} onChange={() => toggleGive(pick)} />
                  Pick #{pick} (Rd {Math.ceil(pick / teams)})
                </label>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Team B</label>
            <select
              value={teamB}
              onChange={(e) => handleTeamBChange(Number(e.target.value))}
              className="border border-gray-300 rounded px-2 py-1 text-sm w-full"
            >
              {allTeams
                .filter((t) => t !== teamA)
                .map((t) => (
                  <option key={t} value={t}>
                    Team {t}
                  </option>
                ))}
            </select>
            <h3 className="text-sm font-medium mt-2 mb-1">Gives</h3>
            <div className="space-y-1 max-h-48 overflow-auto">
              {teamBPicks.map((pick) => (
                <label key={pick} className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" checked={selectedReceives.has(pick)} onChange={() => toggleReceive(pick)} />
                  Pick #{pick} (Rd {Math.ceil(pick / teams)})
                </label>
              ))}
            </div>
          </div>
        </div>

        {evaluation && (
          <div
            className={`mb-4 p-3 rounded text-sm ${
              evaluation.netValue >= 0 ? "bg-green-50 border border-green-200" : "bg-red-50 border border-red-200"
            }`}
          >
            <div className="grid grid-cols-3 gap-2 mb-2">
              <div>
                <span className="text-xs text-gray-500">Team A gives</span>
                <p className="font-medium">{evaluation.givesValue.toFixed(1)}</p>
              </div>
              <div>
                <span className="text-xs text-gray-500">Team B gives</span>
                <p className="font-medium">{evaluation.receivesValue.toFixed(1)}</p>
              </div>
              <div>
                <span className="text-xs text-gray-500">Net (Team A)</span>
                <p className={`font-medium ${evaluation.netValue >= 0 ? "text-green-700" : "text-red-700"}`}>
                  {evaluation.netValue >= 0 ? "+" : ""}
                  {evaluation.netValue.toFixed(1)}
                </p>
              </div>
            </div>
            <div className="mb-2">
              {evaluation.givesDetail.map((d) => (
                <span key={d.pickNumber} className="text-xs text-gray-600 mr-2">
                  Pick #{d.pickNumber}: {d.value.toFixed(1)}
                </span>
              ))}
              {evaluation.receivesDetail.map((d) => (
                <span key={d.pickNumber} className="text-xs text-gray-600 mr-2">
                  Pick #{d.pickNumber}: {d.value.toFixed(1)}
                </span>
              ))}
            </div>
            <p className={`text-xs font-medium ${evaluation.netValue >= 0 ? "text-green-700" : "text-red-700"}`}>
              {evaluation.recommendation}
            </p>
          </div>
        )}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 text-sm border border-gray-300 rounded hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleEvaluate}
            disabled={!hasSelection || evaluating}
            className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
          >
            {evaluating && <Spinner className="h-3 w-3" />}
            Evaluate
          </button>
          <button
            type="button"
            onClick={handleExecute}
            disabled={!hasSelection || executing}
            className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
          >
            {executing && <Spinner className="h-3 w-3" />}
            Execute Trade
          </button>
        </div>
      </div>
    </div>
  );
}
