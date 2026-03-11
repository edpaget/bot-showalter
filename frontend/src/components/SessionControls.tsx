import { useState } from "react";
import type { DraftSessionSummaryType, DraftStateType } from "../generated/graphql";
import { Spinner } from "./Spinner";

interface SessionControlsProps {
  sessionActive: boolean;
  state: DraftStateType | null;
  sessions: DraftSessionSummaryType[];
  onStart: (config: { season: number; teams: number; format: string; userTeam: number; budget?: number }) => void;
  onResume: (sessionId: number) => void;
  onUndo: () => void;
  onEnd: () => void;
  loading?: boolean;
  undoing?: boolean;
}

export function SessionControls({
  sessionActive,
  state,
  sessions,
  onStart,
  onResume,
  onUndo,
  onEnd,
  loading,
  undoing,
}: SessionControlsProps) {
  const [season, setSeason] = useState(2026);
  const [teams, setTeams] = useState(12);
  const [format, setFormat] = useState("snake");
  const [userTeam, setUserTeam] = useState(1);
  const [budget, setBudget] = useState(260);

  if (sessionActive && state) {
    return (
      <div className="flex items-center gap-3 p-3 bg-gray-50 rounded border border-gray-200">
        <span className="text-sm font-semibold">Pick #{state.currentPick}</span>
        <span className="text-xs text-gray-500">
          {state.format} · {state.teams} teams
        </span>
        {state.budgetRemaining != null && (
          <span className="text-xs text-gray-500">Budget: ${state.budgetRemaining}</span>
        )}
        <div className="flex-1" />
        <button
          type="button"
          onClick={onUndo}
          disabled={undoing || state.picks.length === 0}
          className="px-3 py-1 text-sm bg-amber-500 text-white rounded hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
        >
          {undoing && <Spinner className="h-3 w-3" />}
          Undo
        </button>
        <button
          type="button"
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
          type="button"
          disabled={loading}
          onClick={() =>
            onStart({
              season,
              teams,
              format,
              userTeam,
              budget: format === "auction" ? budget : undefined,
            })
          }
          className="px-4 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
        >
          {loading && <Spinner className="h-3 w-3" />}
          {loading ? "Starting…" : "Start Draft"}
        </button>
      </div>

      {resumableSessions.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-xs text-gray-500 mb-1">Resume session:</p>
          <div className="flex flex-wrap gap-2">
            {resumableSessions.map((s) => (
              <button
                type="button"
                key={s.id}
                disabled={loading}
                onClick={() => onResume(s.id)}
                className="px-3 py-1 text-xs bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
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
