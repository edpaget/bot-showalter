import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type {
  ArbitrageReportType,
  CategoryBalanceType,
  DraftPickType,
  DraftStateType,
  KeeperInfoType,
  PickResultFieldsFragment,
  RecommendationType,
  RosterSlotType,
} from "../generated/graphql";

export type { KeeperInfoType as KeeperInfo } from "../generated/graphql";

interface DraftSessionContextValue {
  sessionId: number | null;
  state: DraftStateType | null;
  recommendations: RecommendationType[];
  roster: DraftPickType[];
  needs: RosterSlotType[];
  balance: CategoryBalanceType[];
  arbitrage: ArbitrageReportType | null;
  keepers: KeeperInfoType[];
  draftedPlayerIds: Set<number>;
  setSessionId: (id: number | null) => void;
  setState: (state: DraftStateType | null) => void;
  setRecommendations: (recs: RecommendationType[]) => void;
  setRoster: (roster: DraftPickType[]) => void;
  setNeeds: (needs: RosterSlotType[]) => void;
  setBalance: (balance: CategoryBalanceType[]) => void;
  setArbitrage: (arbitrage: ArbitrageReportType | null) => void;
  setKeepers: (keepers: KeeperInfoType[]) => void;
  applyPickResult: (result: PickResultFieldsFragment) => void;
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
  const [arbitrage, setArbitrage] = useState<ArbitrageReportType | null>(null);
  const [keepers, setKeepers] = useState<KeeperInfoType[]>([]);

  const draftedPlayerIds = useMemo(() => {
    if (!state) return new Set<number>();
    return new Set(state.picks.map((p) => p.playerId));
  }, [state]);

  const applyPickResult = useCallback((result: PickResultFieldsFragment) => {
    setState(result.state as DraftStateType);
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
