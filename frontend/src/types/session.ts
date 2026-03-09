export interface DraftPick {
  pickNumber: number;
  team: number;
  playerId: number;
  playerName: string;
  position: string;
  price: number | null;
}

export interface DraftState {
  sessionId: number;
  currentPick: number;
  picks: DraftPick[];
  format: string;
  teams: number;
  userTeam: number;
  budgetRemaining: number | null;
}

export interface DraftSessionSummary {
  id: number;
  league: string;
  season: number;
  teams: number;
  format: string;
  userTeam: number;
  status: string;
  pickCount: number;
  createdAt: string;
  updatedAt: string;
  system: string;
  version: string;
}

export interface Recommendation {
  playerId: number;
  playerName: string;
  position: string;
  value: number;
  score: number;
  reason: string;
}

export interface RosterSlot {
  position: string;
  remaining: number;
}

export interface CategoryBalance {
  category: string;
  projectedValue: number;
  leagueRankEstimate: number;
  strength: string;
}

export interface PickResult {
  pick: DraftPick;
  state: DraftState;
  recommendations: Recommendation[];
  roster: DraftPick[];
  needs: RosterSlot[];
}

export interface YahooPollStatus {
  active: boolean;
  lastPollAt: string | null;
  picksIngested: number;
}
