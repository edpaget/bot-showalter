import { useLazyQuery } from "@apollo/client";
import { useEffect, useState } from "react";
import type { DraftSessionSummaryType, DraftStateType, YahooDraftSetupQuery } from "../generated/graphql";
import { YAHOO_DRAFT_SETUP_QUERY } from "../graphql/queries";
import { Spinner } from "./Spinner";

interface SessionControlsProps {
  sessionActive: boolean;
  state: DraftStateType | null;
  sessions: DraftSessionSummaryType[];
  onStart: (config: {
    season: number;
    teams: number;
    format: string;
    userTeam: number;
    budget?: number;
    keeperPlayerIds?: number[];
    leagueKey?: string;
    teamNames?: Record<string, string>;
  }) => void;
  onResume: (sessionId: number) => void;
  onUndo: () => void;
  onEnd: () => void;
  loading?: boolean;
  undoing?: boolean;
  onTrade?: () => void;
  tradeDisabled?: boolean;
  isSnakeFormat?: boolean;
  yahooLeague?: { leagueKey: string; leagueName: string; season: number } | null;
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
  onTrade,
  tradeDisabled,
  isSnakeFormat,
  yahooLeague,
}: SessionControlsProps) {
  const [season, setSeason] = useState(2026);
  const [teams, setTeams] = useState(12);
  const [format, setFormat] = useState("snake");
  const [userTeam, setUserTeam] = useState(1);
  const [budget, setBudget] = useState(260);
  const [keeperPlayerIds, setKeeperPlayerIds] = useState<number[]>([]);
  const [prefilled, setPrefilled] = useState(false);
  const [prefillError, setPrefillError] = useState(false);
  const [prefillTeamNames, setPrefillTeamNames] = useState<Record<string, string> | undefined>(undefined);

  const [fetchSetup] = useLazyQuery<YahooDraftSetupQuery>(YAHOO_DRAFT_SETUP_QUERY);

  useEffect(() => {
    if (!yahooLeague || sessionActive) return;
    fetchSetup({ variables: { leagueKey: yahooLeague.leagueKey, season: yahooLeague.season } }).then(
      ({ data, error }) => {
        if (error || !data) {
          setPrefillError(true);
          return;
        }
        const setup = data.yahooDraftSetup;
        setSeason(yahooLeague.season);
        setTeams(setup.numTeams);
        const fmt = setup.draftFormat === "auction" ? "auction" : "snake";
        setFormat(fmt);
        setUserTeam(setup.userTeamId);
        setKeeperPlayerIds(setup.keeperPlayerIds);
        if (setup.teamNames) {
          setPrefillTeamNames(setup.teamNames as Record<string, string>);
        }
        setPrefilled(true);
      },
    );
  }, [yahooLeague, sessionActive, fetchSetup]);

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
        {isSnakeFormat && onTrade && (
          <button
            type="button"
            onClick={onTrade}
            disabled={tradeDisabled}
            className="px-3 py-1 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Trade Picks
          </button>
        )}
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
      {prefilled && yahooLeague && (
        <p className="text-xs text-blue-600 mb-2">Prefilled from Yahoo: {yahooLeague.leagueName}</p>
      )}
      {prefillError && (
        <p className="text-xs text-amber-600 mb-2">Could not load Yahoo settings — enter values manually</p>
      )}
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
              keeperPlayerIds: keeperPlayerIds.length > 0 ? keeperPlayerIds : undefined,
              leagueKey: yahooLeague?.leagueKey,
              teamNames: prefillTeamNames,
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
