import type { Position } from "./position";

export interface DraftPick {
  pickNumber: number;
  team: number;
  playerId: number;
  playerName: string;
  position: Position;
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
  keeperCount: number;
}

export interface KeeperInfo {
  playerId: number;
  playerName: string;
  position: string;
  teamName: string;
  cost: number | null;
  value: number;
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
  position: Position;
  value: number;
  score: number;
  reason: string;
}

export interface RosterSlot {
  position: Position;
  remaining: number;
}

export interface CategoryBalance {
  category: string;
  projectedValue: number;
  leagueRankEstimate: number;
  strength: string;
}

export interface FallingPlayer {
  playerId: number;
  playerName: string;
  position: string;
  adp: number;
  currentPick: number;
  picksPastAdp: number;
  value: number;
  valueRank: number;
  arbitrageScore: number;
}

export interface ReachPick {
  playerId: number;
  playerName: string;
  position: string;
  adp: number;
  pickNumber: number;
  picksAheadOfAdp: number;
  drafterTeam: number;
}

export interface ArbitrageReport {
  currentPick: number;
  falling: FallingPlayer[];
  reaches: ReachPick[];
}

export interface PickResult {
  pick: DraftPick;
  state: DraftState;
  recommendations: Recommendation[];
  roster: DraftPick[];
  needs: RosterSlot[];
  arbitrage: ArbitrageReport | null;
}

export interface YahooPollStatus {
  active: boolean;
  lastPollAt: string | null;
  picksIngested: number;
}
