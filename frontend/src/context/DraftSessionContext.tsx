import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type {
  ArbitrageReportType,
  CategoryBalanceType,
  DraftPickType,
  DraftStateType,
  PickResultFieldsFragment,
  RecommendationType,
  RosterSlotType,
} from "../generated/graphql";

// KeeperInfo is not yet in codegen — defined here until the schema is updated.
export interface KeeperInfo {
  playerId: number;
  playerName: string;
  position: string;
  teamName: string;
  cost: number | null;
  value: number;
}

// Extend DraftStateType with keeperCount (added on main after codegen was generated).
export type DraftState = DraftStateType & { keeperCount: number };

interface DraftSessionContextValue {
  sessionId: number | null;
  state: DraftState | null;
  recommendations: RecommendationType[];
  roster: DraftPickType[];
  needs: RosterSlotType[];
  balance: CategoryBalanceType[];
  arbitrage: ArbitrageReportType | null;
  keepers: KeeperInfo[];
  draftedPlayerIds: Set<number>;
  setSessionId: (id: number | null) => void;
  setState: (state: DraftState | null) => void;
  setRecommendations: (recs: RecommendationType[]) => void;
  setRoster: (roster: DraftPickType[]) => void;
  setNeeds: (needs: RosterSlotType[]) => void;
  setBalance: (balance: CategoryBalanceType[]) => void;
  setArbitrage: (arbitrage: ArbitrageReportType | null) => void;
  setKeepers: (keepers: KeeperInfo[]) => void;
  applyPickResult: (result: PickResultFieldsFragment) => void;
  clearSession: () => void;
}

const DraftSessionContext = createContext<DraftSessionContextValue | null>(null);

export function DraftSessionProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [state, setState] = useState<DraftState | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationType[]>([]);
  const [roster, setRoster] = useState<DraftPickType[]>([]);
  const [needs, setNeeds] = useState<RosterSlotType[]>([]);
  const [balance, setBalance] = useState<CategoryBalanceType[]>([]);
  const [arbitrage, setArbitrage] = useState<ArbitrageReportType | null>(null);
  const [keepers, setKeepers] = useState<KeeperInfo[]>([]);

  const draftedPlayerIds = useMemo(() => {
    if (!state) return new Set<number>();
    return new Set(state.picks.map((p) => p.playerId));
  }, [state]);

  const applyPickResult = useCallback((result: PickResultFieldsFragment) => {
    setState(result.state as DraftState);
    setRecommendations(result.recommendations);
    setRoster(result.roster);
    setNeeds(result.needs);
    setArbitrage(result.arbitrage);
  }, []);

  const clearSession = useCallback(() => {
    setSessionId(null);
    setState(null);
    setRecommendations([]);
    setRoster([]);
    setNeeds([]);
    setBalance([]);
    setArbitrage(null);
    setKeepers([]);
  }, []);

  const value = useMemo(
    () => ({
      sessionId,
      state,
      recommendations,
      roster,
      needs,
      balance,
      arbitrage,
      keepers,
      draftedPlayerIds,
      setSessionId,
      setState,
      setRecommendations,
      setRoster,
      setNeeds,
      setBalance,
      setArbitrage,
      setKeepers,
      applyPickResult,
      clearSession,
    }),
    [
      sessionId,
      state,
      recommendations,
      roster,
      needs,
      balance,
      arbitrage,
      keepers,
      draftedPlayerIds,
      applyPickResult,
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
