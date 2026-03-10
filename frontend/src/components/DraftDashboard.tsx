import { useCallback, useEffect } from "react";
import { useMutation, useQuery, useSubscription, useLazyQuery } from "@apollo/client";
import { useDraftSession } from "../context/DraftSessionContext";
import { SESSIONS_QUERY, BALANCE_QUERY, SESSION_QUERY, RECOMMENDATIONS_QUERY, ROSTER_QUERY, NEEDS_QUERY } from "../graphql/queries";
import { START_SESSION, PICK, UNDO, END_SESSION } from "../graphql/mutations";
import { DRAFT_EVENTS_SUBSCRIPTION } from "../graphql/subscriptions";
import { DraftBoardTable } from "./DraftBoardTable";
import { RecommendationPanel } from "./RecommendationPanel";
import { RosterPanel } from "./RosterPanel";
import { NeedsPanel } from "./NeedsPanel";
import { CategoryBalancePanel } from "./CategoryBalancePanel";
import { ArbitragePanel } from "./ArbitragePanel";
import { PickLogPanel } from "./PickLogPanel";
import { SessionControls } from "./SessionControls";
import type { DraftSessionSummary, PickResult, DraftState, CategoryBalance, Recommendation, DraftPick, RosterSlot } from "../types/session";

export function DraftDashboard({ season = 2026 }: { season?: number }) {
  const ctx = useDraftSession();
  const sessionActive = ctx.sessionId != null && ctx.state != null;

  const { data: sessionsData } = useQuery<{ sessions: DraftSessionSummary[] }>(SESSIONS_QUERY, {
    variables: { status: "active" },
    skip: sessionActive,
  });

  const { data: balanceData } = useQuery<{ balance: CategoryBalance[] }>(BALANCE_QUERY, {
    variables: { sessionId: ctx.sessionId },
    skip: !sessionActive,
  });

  useEffect(() => {
    if (balanceData?.balance) {
      ctx.setBalance(balanceData.balance);
    }
  }, [balanceData]); // eslint-disable-line react-hooks/exhaustive-deps

  const [fetchSession] = useLazyQuery<{ session: DraftState }>(SESSION_QUERY);
  const [fetchRecs] = useLazyQuery<{ recommendations: Recommendation[] }>(RECOMMENDATIONS_QUERY);
  const [fetchRoster] = useLazyQuery<{ roster: DraftPick[] }>(ROSTER_QUERY);
  const [fetchNeeds] = useLazyQuery<{ needs: RosterSlot[] }>(NEEDS_QUERY);
  const [startSession] = useMutation<{ startSession: DraftState }>(START_SESSION);
  const [pickMutation] = useMutation<{ pick: PickResult }>(PICK, {
    refetchQueries: [{ query: BALANCE_QUERY, variables: { sessionId: ctx.sessionId } }],
  });
  const [undoMutation] = useMutation<{ undo: PickResult }>(UNDO, {
    refetchQueries: [{ query: BALANCE_QUERY, variables: { sessionId: ctx.sessionId } }],
  });
  const [endSession] = useMutation(END_SESSION);

  useSubscription(DRAFT_EVENTS_SUBSCRIPTION, {
    variables: { sessionId: ctx.sessionId },
    skip: !sessionActive,
    onData: ({ data: subData }) => {
      const event = subData.data?.draftEvents;
      if (!event || !ctx.state) return;

      if (event.__typename === "PickEvent") {
        const pick = event.pick;
        ctx.setState({
          ...ctx.state,
          currentPick: ctx.state.currentPick + 1,
          picks: [...ctx.state.picks, pick],
        });
      } else if (event.__typename === "UndoEvent") {
        ctx.setState({
          ...ctx.state,
          currentPick: Math.max(1, ctx.state.currentPick - 1),
          picks: ctx.state.picks.slice(0, -1),
        });
      }
    },
  });

  const handleStart = useCallback(
    async (config: { season: number; teams: number; format: string; userTeam: number; budget?: number }) => {
      const result = await startSession({
        variables: {
          season: config.season,
          teams: config.teams,
          format: config.format,
          userTeam: config.userTeam,
          budget: config.budget,
        },
      });
      if (result.data) {
        const state = result.data.startSession;
        ctx.setSessionId(state.sessionId);
        ctx.setState(state);
      }
    },
    [startSession, ctx],
  );

  const handleResume = useCallback(
    async (sessionId: number) => {
      const [sessionRes, recsRes, rosterRes, needsRes] = await Promise.all([
        fetchSession({ variables: { sessionId } }),
        fetchRecs({ variables: { sessionId, limit: 10 } }),
        fetchRoster({ variables: { sessionId } }),
        fetchNeeds({ variables: { sessionId } }),
      ]);
      if (sessionRes.data) {
        ctx.setSessionId(sessionId);
        ctx.setState(sessionRes.data.session);
        ctx.setRecommendations(recsRes.data?.recommendations ?? []);
        ctx.setRoster(rosterRes.data?.roster ?? []);
        ctx.setNeeds(needsRes.data?.needs ?? []);
      }
    },
    [ctx, fetchSession, fetchRecs, fetchRoster, fetchNeeds],
  );

  const handleDraft = useCallback(
    async (playerId: number, position: string) => {
      if (!ctx.sessionId) return;
      const result = await pickMutation({
        variables: { sessionId: ctx.sessionId, playerId, position },
      });
      if (result.data) {
        ctx.applyPickResult(result.data.pick);
      }
    },
    [ctx, pickMutation],
  );

  const handleUndo = useCallback(async () => {
    if (!ctx.sessionId) return;
    const result = await undoMutation({
      variables: { sessionId: ctx.sessionId },
    });
    if (result.data) {
      ctx.applyPickResult(result.data.undo);
    }
  }, [ctx, undoMutation]);

  const handleEnd = useCallback(async () => {
    if (!ctx.sessionId) return;
    await endSession({ variables: { sessionId: ctx.sessionId } });
    ctx.clearSession();
  }, [ctx, endSession]);

  return (
    <div className="flex flex-col gap-3 h-screen p-3">
      <SessionControls
        sessionActive={sessionActive}
        state={ctx.state}
        sessions={sessionsData?.sessions ?? []}
        onStart={handleStart}
        onResume={handleResume}
        onUndo={handleUndo}
        onEnd={handleEnd}
      />

      <div className="flex gap-3 flex-1 min-h-0">
        <div className="flex-1 min-w-0">
          <DraftBoardTable
            season={season}
            draftedPlayerIds={sessionActive ? ctx.draftedPlayerIds : undefined}
            onDraft={sessionActive ? handleDraft : undefined}
            sessionActive={sessionActive}
          />
        </div>

        {sessionActive && (
          <div className="w-80 flex-shrink-0 flex flex-col gap-3 overflow-auto">
            <RecommendationPanel
              recommendations={ctx.recommendations}
              onDraft={handleDraft}
              sessionActive
            />
            <ArbitragePanel
              arbitrage={ctx.arbitrage}
              sessionId={ctx.sessionId!}
              onDraft={handleDraft}
            />
            <RosterPanel
              roster={ctx.roster}
              needs={ctx.needs}
              budgetRemaining={ctx.state?.budgetRemaining ?? null}
              format={ctx.state?.format ?? "snake"}
            />
            <NeedsPanel needs={ctx.needs} />
            <CategoryBalancePanel balance={ctx.balance} />
          </div>
        )}
      </div>

      {sessionActive && ctx.state && <PickLogPanel picks={ctx.state.picks} />}
    </div>
  );
}
