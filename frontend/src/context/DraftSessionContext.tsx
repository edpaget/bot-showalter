import type { ReactNode } from "react";
import { createContext, startTransition, useCallback, useContext, useMemo, useState } from "react";
import type {
  ArbitrageReportType,
  CategoryBalanceType,
  CategoryNeedType,
  DraftPickType,
  DraftStateType,
  KeeperInfoType,
  PickResultFieldsFragment,
  RecommendationType,
  RosterSlotType,
} from "../generated/graphql";

export type { KeeperInfoType as KeeperInfo } from "../generated/graphql";

const PITCHER_POSITIONS = new Set(["SP", "RP", "P"]);

function playerKey(playerId: number, playerType: string): string {
  return `${playerId}-${playerType}`;
}

function pickPlayerType(position: string): string {
  return PITCHER_POSITIONS.has(position) ? "pitcher" : "batter";
}

interface DraftSessionContextValue {
  sessionId: number | null;
  state: DraftStateType | null;
  recommendations: RecommendationType[];
  roster: DraftPickType[];
  needs: RosterSlotType[];
  balance: CategoryBalanceType[];
  categoryNeeds: CategoryNeedType[];
  arbitrage: ArbitrageReportType | null;
  keepers: KeeperInfoType[];
  teamNames: Record<number, string>;
  draftedPlayerKeys: Set<string>;
  getTeamName: (id: number) => string;
  setSessionId: (id: number | null) => void;
  setState: (state: DraftStateType | null) => void;
  setRecommendations: (recs: RecommendationType[]) => void;
  setRoster: (roster: DraftPickType[]) => void;
  setNeeds: (needs: RosterSlotType[]) => void;
  setBalance: (balance: CategoryBalanceType[]) => void;
  setCategoryNeeds: (needs: CategoryNeedType[]) => void;
  setArbitrage: (arbitrage: ArbitrageReportType | null) => void;
  setKeepers: (keepers: KeeperInfoType[]) => void;
  setTeamNames: (names: Record<number, string>) => void;
  applyPickResult: (result: PickResultFieldsFragment) => void;
  addOptimisticPick: (playerId: number, position: string, playerType: string) => void;
  clearSession: () => void;
}

const DraftSessionContext = createContext<DraftSessionContextValue | null>(null);

export function DraftSessionProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [state, setState] = useState<DraftStateType | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationType[]>([]);
  const [roster, setRoster] = useState<DraftPickType[]>([]);
  const [needs, setNeeds] = useState<RosterSlotType[]>([]);
  const [balance, setBalance] = useState<CategoryBalanceType[]>([]);
  const [categoryNeeds, setCategoryNeeds] = useState<CategoryNeedType[]>([]);
  const [arbitrage, setArbitrage] = useState<ArbitrageReportType | null>(null);
  const [keepers, setKeepers] = useState<KeeperInfoType[]>([]);
  const [teamNames, setTeamNames] = useState<Record<number, string>>({});

  const getTeamName = useCallback((id: number): string => teamNames[id] ?? `Team ${id}`, [teamNames]);

  const draftedPlayerKeys = useMemo(() => {
    if (!state) return new Set<string>();
    const keys = new Set(state.picks.map((p) => playerKey(p.playerId, pickPlayerType(p.position))));
    for (const k of keepers) {
      if (k.playerType) {
        keys.add(playerKey(k.playerId, k.playerType));
      } else {
        keys.add(playerKey(k.playerId, "batter"));
        keys.add(playerKey(k.playerId, "pitcher"));
      }
    }
    return keys;
  }, [state, keepers]);

  const applyPickResult = useCallback((result: PickResultFieldsFragment) => {
    // Use startTransition so the optimistic state stays visible while React
    // processes the full result update in the background.
    startTransition(() => {
      setState(result.state as DraftStateType);
      setRecommendations(result.recommendations);
      setRoster(result.roster);
      setNeeds(result.needs);
      setArbitrage(result.arbitrage);
      setBalance(result.balance);
      setCategoryNeeds(result.categoryNeeds);
    });
  }, []);

  const addOptimisticPick = useCallback(
    (playerId: number, position: string, playerType: string) => {
      if (!state) return;
      // Immediately mark the player as drafted and advance the pick counter.
      // Placeholder fields (playerName, position, etc.) get replaced when
      // the real mutation response arrives via applyPickResult.
      setState({
        ...state,
        currentPick: state.currentPick + 1,
        picks: [
          ...state.picks,
          {
            __typename: "DraftPickType" as const,
            playerId,
            playerName: "…",
            position: position as DraftPickType["position"],
            playerType,
            team: 0,
            pickNumber: state.currentPick,
            price: null,
          },
        ],
      });
      // Filter the drafted player out of recommendations (match both id and type
      // so drafting batter-Ohtani doesn't remove pitcher-Ohtani from recs).
      setRecommendations((prev) => prev.filter((r) => !(r.playerId === playerId && r.playerType === playerType)));
    },
    [state],
  );

  const clearSession = useCallback(() => {
    setSessionId(null);
    setState(null);
    setRecommendations([]);
    setRoster([]);
    setNeeds([]);
    setBalance([]);
    setCategoryNeeds([]);
    setArbitrage(null);
    setKeepers([]);
    setTeamNames({});
  }, []);

  const value = useMemo(
    () => ({
      sessionId,
      state,
      recommendations,
      roster,
      needs,
      balance,
      categoryNeeds,
      arbitrage,
      keepers,
      teamNames,
      draftedPlayerKeys,
      getTeamName,
      setSessionId,
      setState,
      setRecommendations,
      setRoster,
      setNeeds,
      setBalance,
      setCategoryNeeds,
      setArbitrage,
      setKeepers,
      setTeamNames,
      applyPickResult,
      addOptimisticPick,
      clearSession,
    }),
    [
      sessionId,
      state,
      recommendations,
      roster,
      needs,
      balance,
      categoryNeeds,
      arbitrage,
      keepers,
      teamNames,
      draftedPlayerKeys,
      getTeamName,
      applyPickResult,
      addOptimisticPick,
      clearSession,
    ],
  );

  return <DraftSessionContext.Provider value={value}>{children}</DraftSessionContext.Provider>;
}

export function useDraftSession(): DraftSessionContextValue {
  const ctx = useContext(DraftSessionContext);
  if (!ctx) {
    throw new Error("useDraftSession must be used within a DraftSessionProvider");
  }
  return ctx;
}
