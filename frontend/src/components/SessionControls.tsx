import { useState } from "react";
import type { DraftSessionSummary, DraftState } from "../types/session";

interface SessionControlsProps {
  sessionActive: boolean;
  state: DraftState | null;
  sessions: DraftSessionSummary[];
  onStart: (config: { season: number; teams: number; format: string; userTeam: number; budget?: number }) => void;
  onResume: (sessionId: number) => void;
  onUndo: () => void;
  onEnd: () => void;
}

export function SessionControls({
  sessionActive,
  state,
  sessions,
  onStart,
  onResume,
  onUndo,
  onEnd,
}: SessionControlsProps) {
  const [season, setSeason] = useState(2026);
  const [teams, setTeams] = useState(12);
  const [format, setFormat] = useState("snake");
  const [userTeam, setUserTeam] = useState(1);
  const [budget, setBudget] = useState(260);

  if (sessionActive && state) {
    return (
      <div className="flex items-center gap-3 p-3 bg-gray-50 rounded border border-gray-200">
        <span className="text-sm font-semibold">
          Pick #{state.currentPick}
        </span>
        <span className="text-xs text-gray-500">
          {state.format} · {state.teams} teams
        </span>
        {state.budgetRemaining != null && (
          <span className="text-xs text-gray-500">Budget: ${state.budgetRemaining}</span>
        )}
        <div className="flex-1" />
        <button
          onClick={onUndo}
          disabled={state.picks.length === 0}
          className="px-3 py-1 text-sm bg-amber-500 text-white rounded hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Undo
        </button>
        <button
          onClick={onEnd}
          className="px-3 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700"
        >
          End Session
        </button>
      </div>
    );
  }

  const resumableSessions = sessions.filter((s) => s.status === "active");

  return (
    <div className="p-3 bg-gray-50 rounded border border-gray-200">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Season</label>
          <input
            type="number"
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
            className="border border-gray-300 rounded px-2 py-1 text-sm w-20"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Teams</label>
          <input
            type="number"
            value={teams}
            onChange={(e) => setTeams(Number(e.target.value))}
            className="border border-gray-300 rounded px-2 py-1 text-sm w-16"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Format</label>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value)}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
          >
            <option value="snake">Snake</option>
            <option value="auction">Auction</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Your Team</label>
          <input
            type="number"
            value={userTeam}
            onChange={(e) => setUserTeam(Number(e.target.value))}
            className="border border-gray-300 rounded px-2 py-1 text-sm w-16"
          />
        </div>
        {format === "auction" && (
          <div>
            <label className="block text-xs text-gray-500 mb-1">Budget</label>
            <input
              type="number"
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="border border-gray-300 rounded px-2 py-1 text-sm w-20"
            />
          </div>
        )}
        <button
          onClick={() =>
            onStart({
              season,
              teams,
              format,
              userTeam,
              budget: format === "auction" ? budget : undefined,
            })
          }
          className="px-4 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Start Draft
        </button>
      </div>

      {resumableSessions.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-xs text-gray-500 mb-1">Resume session:</p>
          <div className="flex flex-wrap gap-2">
            {resumableSessions.map((s) => (
              <button
                key={s.id}
                onClick={() => onResume(s.id)}
                className="px-3 py-1 text-xs bg-white border border-gray-300 rounded hover:bg-gray-100"
              >
                #{s.id} — {s.season} {s.format} ({s.pickCount} picks)
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
