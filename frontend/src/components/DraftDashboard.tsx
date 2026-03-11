import { useLazyQuery, useMutation, useQuery, useSubscription } from "@apollo/client";
import { useCallback, useEffect } from "react";
import { useDraftSession } from "../context/DraftSessionContext";
import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import type {
  BalanceQuery,
  CategoryNeedsQuery,
  DraftEventsSubscription,
  PickMutation,
  SessionQuery,
  SessionsQuery,
  StartSessionMutation,
  UndoMutation,
} from "../generated/graphql";
import { END_SESSION, PICK, START_SESSION, UNDO } from "../graphql/mutations";
import {
  BALANCE_QUERY,
  CATEGORY_NEEDS_QUERY,
  KEEPERS_QUERY,
  NEEDS_QUERY,
  RECOMMENDATIONS_QUERY,
  ROSTER_QUERY,
  SESSION_QUERY,
  SESSIONS_QUERY,
} from "../graphql/queries";
import { DRAFT_EVENTS_SUBSCRIPTION } from "../graphql/subscriptions";
import { ArbitragePanel } from "./ArbitragePanel";
import { CategoryBalancePanel } from "./CategoryBalancePanel";
import { CategoryNeedsPanel } from "./CategoryNeedsPanel";
import { DraftBoardTable } from "./DraftBoardTable";
import { KeeperPanel } from "./KeeperPanel";
import { NeedsPanel } from "./NeedsPanel";
import { PickLogPanel } from "./PickLogPanel";
import { RecommendationPanel } from "./RecommendationPanel";
import { RosterPanel } from "./RosterPanel";
import { SessionControls } from "./SessionControls";

export function DraftDashboard({ season = 2026 }: { season?: number }) {
  const ctx = useDraftSession();
  const { openPlayer } = usePlayerDrawer();
  const sessionActive = ctx.sessionId != null && ctx.state != null;

  const { data: sessionsData } = useQuery<SessionsQuery>(SESSIONS_QUERY, {
    variables: { status: "active" },
    skip: sessionActive,
  });

  const { data: balanceData } = useQuery<BalanceQuery>(BALANCE_QUERY, {
    variables: { sessionId: ctx.sessionId },
    skip: !sessionActive,
  });

  const { data: categoryNeedsData } = useQuery<CategoryNeedsQuery>(CATEGORY_NEEDS_QUERY, {
    variables: { sessionId: ctx.sessionId },
    skip: !sessionActive,
  });

  useEffect(() => {
    if (balanceData?.balance) {
      ctx.setBalance(balanceData.balance);
    }
  }, [balanceData, ctx.setBalance]);

  useEffect(() => {
    if (categoryNeedsData?.categoryNeeds) {
      ctx.setCategoryNeeds(categoryNeedsData.categoryNeeds);
    }
  }, [categoryNeedsData, ctx.setCategoryNeeds]);

  const [fetchSession] = useLazyQuery<SessionQuery>(SESSION_QUERY);
  const [fetchRecs] = useLazyQuery(RECOMMENDATIONS_QUERY);
  const [fetchRoster] = useLazyQuery(ROSTER_QUERY);
  const [fetchNeeds] = useLazyQuery(NEEDS_QUERY);
  const [fetchKeepers] = useLazyQuery(KEEPERS_QUERY);
  const [startSession, { loading: starting }] = useMutation<StartSessionMutation>(START_SESSION);
  const [pickMutation, { loading: picking }] = useMutation<PickMutation>(PICK);
  const [undoMutation, { loading: undoing }] = useMutation<UndoMutation>(UNDO);
  const [endSession] = useMutation(END_SESSION);

  useSubscription<DraftEventsSubscription>(DRAFT_EVENTS_SUBSCRIPTION, {
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
        // Fetch keepers if this is a keeper session
        if (state.keeperCount > 0) {
          const keepersRes = await fetchKeepers({ variables: { sessionId: state.sessionId } });
          ctx.setKeepers(keepersRes.data?.keepers ?? []);
        }
      }
    },
    [startSession, ctx, fetchKeepers],
  );

  const handleResume = useCallback(
    async (sessionId: number) => {
      const [sessionRes, recsRes, rosterRes, needsRes, keepersRes] = await Promise.all([
        fetchSession({ variables: { sessionId } }),
        fetchRecs({ variables: { sessionId, position: null, limit: 10 } }),
        fetchRoster({ variables: { sessionId, team: null } }),
        fetchNeeds({ variables: { sessionId } }),
        fetchKeepers({ variables: { sessionId } }),
      ]);
      if (sessionRes.data) {
        ctx.setSessionId(sessionId);
        ctx.setState(sessionRes.data.session);
        ctx.setRecommendations(recsRes.data?.recommendations ?? []);
        ctx.setRoster(rosterRes.data?.roster ?? []);
        ctx.setNeeds(needsRes.data?.needs ?? []);
        ctx.setKeepers(keepersRes.data?.keepers ?? []);
      }
    },
    [ctx, fetchSession, fetchRecs, fetchRoster, fetchNeeds, fetchKeepers],
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
        loading={starting}
        undoing={undoing}
      />

      <div className="flex gap-3 flex-1 min-h-0">
        <div className="flex-1 min-w-0">
          <DraftBoardTable
            season={season}
            draftedPlayerIds={sessionActive ? ctx.draftedPlayerIds : undefined}
            onDraft={sessionActive ? handleDraft : undefined}
            onPlayerClick={openPlayer}
            sessionActive={sessionActive}
            pickLoading={picking}
          />
        </div>

        {sessionActive && (
          <div className="w-80 flex-shrink-0 flex flex-col gap-3 overflow-auto">
            {ctx.keepers.length > 0 && <KeeperPanel keepers={ctx.keepers} />}
            <RecommendationPanel
              recommendations={ctx.recommendations}
              onDraft={handleDraft}
              onPlayerClick={openPlayer}
              sessionActive
              pickLoading={picking}
            />
            <ArbitragePanel
              arbitrage={ctx.arbitrage}
              sessionId={ctx.sessionId!}
              onDraft={handleDraft}
              pickLoading={picking}
            />
            <RosterPanel
              roster={ctx.roster}
              needs={ctx.needs}
              budgetRemaining={ctx.state?.budgetRemaining ?? null}
              format={ctx.state?.format ?? "snake"}
            />
            <NeedsPanel needs={ctx.needs} />
            <CategoryBalancePanel balance={ctx.balance} />
            <CategoryNeedsPanel needs={ctx.categoryNeeds} onPlayerClick={openPlayer} />
          </div>
        )}
      </div>

      {sessionActive && ctx.state && <PickLogPanel picks={ctx.state.picks} onPlayerClick={openPlayer} />}
    </div>
  );
}
