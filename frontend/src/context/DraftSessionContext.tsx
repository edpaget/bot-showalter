import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type {
  ArbitrageReport,
  CategoryBalance,
  DraftPick,
  DraftState,
  PickResult,
  Recommendation,
  RosterSlot,
} from "../types/session";

interface DraftSessionContextValue {
  sessionId: number | null;
  state: DraftState | null;
  recommendations: Recommendation[];
  roster: DraftPick[];
  needs: RosterSlot[];
  balance: CategoryBalance[];
  arbitrage: ArbitrageReport | null;
  draftedPlayerIds: Set<number>;
  setSessionId: (id: number | null) => void;
  setState: (state: DraftState | null) => void;
  setRecommendations: (recs: Recommendation[]) => void;
  setRoster: (roster: DraftPick[]) => void;
  setNeeds: (needs: RosterSlot[]) => void;
  setBalance: (balance: CategoryBalance[]) => void;
  setArbitrage: (arbitrage: ArbitrageReport | null) => void;
  applyPickResult: (result: PickResult) => void;
  clearSession: () => void;
}

const DraftSessionContext = createContext<DraftSessionContextValue | null>(null);

export function DraftSessionProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [state, setState] = useState<DraftState | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [roster, setRoster] = useState<DraftPick[]>([]);
  const [needs, setNeeds] = useState<RosterSlot[]>([]);
  const [balance, setBalance] = useState<CategoryBalance[]>([]);
  const [arbitrage, setArbitrage] = useState<ArbitrageReport | null>(null);

  const draftedPlayerIds = useMemo(() => {
    if (!state) return new Set<number>();
    return new Set(state.picks.map((p) => p.playerId));
  }, [state]);

  const applyPickResult = useCallback((result: PickResult) => {
    setState(result.state);
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
      draftedPlayerIds,
      setSessionId,
      setState,
      setRecommendations,
      setRoster,
      setNeeds,
      setBalance,
      setArbitrage,
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
