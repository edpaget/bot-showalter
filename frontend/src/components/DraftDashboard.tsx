import { useLazyQuery, useMutation, useQuery, useSubscription } from "@apollo/client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useDraftSession } from "../context/DraftSessionContext";
import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import type {
  BalanceQuery,
  CategoryNeedsQuery,
  DraftEventsSubscription,
  DraftStateType,
  LeagueQuery,
  PickMutation,
  SessionQuery,
  SessionsQuery,
  StartSessionMutation,
  UndoMutation,
  UndoTradeMutation,
  WebConfigQuery,
} from "../generated/graphql";
import { END_SESSION, PICK, START_SESSION, UNDO, UNDO_TRADE } from "../graphql/mutations";
import {
  BALANCE_QUERY,
  CATEGORY_NEEDS_QUERY,
  KEEPERS_QUERY,
  LEAGUE_QUERY,
  NEEDS_QUERY,
  RECOMMENDATIONS_QUERY,
  ROSTER_QUERY,
  SESSION_QUERY,
  SESSIONS_QUERY,
  WEB_CONFIG_QUERY,
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
import { TradeDialog } from "./TradeDialog";

export function DraftDashboard({ season = 2026 }: { season?: number }) {
  const ctx = useDraftSession();
  const { openPlayer } = usePlayerDrawer();
  const sessionActive = ctx.sessionId != null && ctx.state != null;
  const [tradeDialogOpen, setTradeDialogOpen] = useState(false);

  const { data: configData } = useQuery<WebConfigQuery>(WEB_CONFIG_QUERY);
  const yahooLeague = configData?.webConfig?.yahooLeague ?? null;

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

  const { data: leagueData } = useQuery<LeagueQuery>(LEAGUE_QUERY, {
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
  const [undoTradeMutation, { loading: undoingTrade }] = useMutation<UndoTradeMutation>(UNDO_TRADE);

  useSubscription<DraftEventsSubscription>(DRAFT_EVENTS_SUBSCRIPTION, {
    variables: { sessionId: ctx.sessionId },
    skip: !sessionActive,
    onData: ({ data: subData }) => {
      const event = subData.data?.draftEvents;
      if (!event || !ctx.state) return;

      if (event.__typename === "PickEvent") {
        const pick = event.pick;
        // Skip if this player was already added (e.g. by our optimistic update)
        if (ctx.state.picks.some((p) => p.playerId === pick.playerId)) return;
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
      } else if (event.__typename === "TradeEvent") {
        ctx.setState({
          ...ctx.state,
          trades: event.state.trades,
        });
      }
    },
  });

  const handleStart = useCallback(
    async (config: {
      season: number;
      teams: number;
      format: string;
      userTeam: number;
      budget?: number;
      keeperPlayerIds?: number[];
      leagueKey?: string;
      teamNames?: Record<string, string>;
      draftOrder?: number[];
    }) => {
      const result = await startSession({
        variables: {
          season: config.season,
          teams: config.teams,
          format: config.format,
          userTeam: config.userTeam,
          budget: config.budget,
          keeperPlayerIds: config.keeperPlayerIds,
          leagueKey: config.leagueKey,
          teamNames: config.teamNames,
          draftOrder: config.draftOrder,
        },
      });
      if (result.data) {
        const state = result.data.startSession;
        ctx.setSessionId(state.sessionId);
        ctx.setState(state);
        // Populate team names from response
        if (state.teamNames) {
          const parsed: Record<number, string> = {};
          for (const [k, v] of Object.entries(state.teamNames as Record<string, string>)) {
            parsed[Number(k)] = v;
          }
          ctx.setTeamNames(parsed);
        }
        // Fetch keepers, recommendations, roster, and needs for keeper sessions
        if (state.keeperCount > 0) {
          const [keepersRes, recsRes, rosterRes, needsRes] = await Promise.all([
            fetchKeepers({ variables: { sessionId: state.sessionId } }),
            fetchRecs({ variables: { sessionId: state.sessionId, position: null, limit: 10 } }),
            fetchRoster({ variables: { sessionId: state.sessionId, team: null } }),
            fetchNeeds({ variables: { sessionId: state.sessionId } }),
          ]);
          ctx.setKeepers(keepersRes.data?.keepers ?? []);
          ctx.setRecommendations(recsRes.data?.recommendations ?? []);
          ctx.setRoster(rosterRes.data?.roster ?? []);
          ctx.setNeeds(needsRes.data?.needs ?? []);
        }
      }
    },
    [startSession, ctx, fetchKeepers, fetchRecs, fetchRoster, fetchNeeds],
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
        // Restore team names from persisted session
        if (sessionRes.data.session.teamNames) {
          const parsed: Record<number, string> = {};
          for (const [k, v] of Object.entries(sessionRes.data.session.teamNames as Record<string, string>)) {
            parsed[Number(k)] = v;
          }
          ctx.setTeamNames(parsed);
        }
      }
    },
    [ctx, fetchSession, fetchRecs, fetchRoster, fetchNeeds, fetchKeepers],
  );

  const handleDraft = useCallback(
    async (playerId: number, position: string) => {
      if (!ctx.sessionId) return;
      // Optimistically mark player as drafted so the board updates instantly
      ctx.addOptimisticPick(playerId);
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

  const handleTradeComplete = useCallback(
    (state: DraftStateType) => {
      ctx.setState(state);
    },
    [ctx],
  );

  const handleUndoTrade = useCallback(async () => {
    if (!ctx.sessionId) return;
    const result = await undoTradeMutation({
      variables: { sessionId: ctx.sessionId },
    });
    if (result.data) {
      ctx.setState(result.data.undoTrade);
    }
  }, [ctx, undoTradeMutation]);

  const league = leagueData?.league;
  const totalPicks = league
    ? (league.rosterBatters + league.rosterPitchers + league.rosterUtil) * (ctx.state?.teams ?? 0)
    : 0;
  const isSnakeFormat = ctx.state?.format === "snake";
  const hasFuturePicks = ctx.state ? ctx.state.currentPick <= totalPicks : false;
  const userTeamName = ctx.state ? ctx.getTeamName(ctx.state.userTeam) : undefined;
  const userKeepers = useMemo(
    () => (userTeamName ? ctx.keepers.filter((k) => k.teamName === userTeamName) : []),
    [ctx.keepers, userTeamName],
  );

  return (
    <div className="flex flex-col gap-3 h-screen p-3 overflow-hidden">
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
        onTrade={() => setTradeDialogOpen(true)}
        tradeDisabled={!hasFuturePicks || !league}
        isSnakeFormat={isSnakeFormat}
        yahooLeague={yahooLeague}
        getTeamName={ctx.getTeamName}
      />

      <div className="flex gap-3 flex-1 min-h-0">
        <div className="flex-1 min-w-0 min-h-0">
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
            {ctx.keepers.length > 0 && <KeeperPanel keepers={ctx.keepers} userTeamName={userTeamName} />}
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
              keepers={userKeepers}
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

      {sessionActive && ctx.state && (
        <PickLogPanel
          picks={ctx.state.picks}
          trades={ctx.state.trades}
          teams={ctx.state.teams}
          userTeam={ctx.state.userTeam}
          teamNames={ctx.teamNames}
          onPlayerClick={openPlayer}
          onUndoTrade={handleUndoTrade}
          undoingTrade={undoingTrade}
        />
      )}

      {tradeDialogOpen && sessionActive && ctx.state && (
        <TradeDialog
          sessionId={ctx.sessionId!}
          userTeam={ctx.state.userTeam}
          teams={ctx.state.teams}
          currentPick={ctx.state.currentPick}
          totalPicks={totalPicks}
          trades={ctx.state.trades}
          teamNames={ctx.teamNames}
          onTradeComplete={handleTradeComplete}
          onClose={() => setTradeDialogOpen(false)}
        />
      )}
    </div>
  );
}
