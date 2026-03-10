import type { Position } from "./position";

export interface Projection {
  playerName: string;
  system: string;
  version: string;
  sourceType: string;
  playerType: string;
  stats: Record<string, number>;
}

export interface ValuationRow {
  playerName: string;
  system: string;
  version: string;
  projectionSystem: string;
  projectionVersion: string;
  playerType: string;
  position: Position;
  value: number;
  rank: number;
  categoryScores: Record<string, number>;
}

export interface ADPReportRow {
  playerId: number;
  playerName: string;
  playerType: string;
  position: Position;
  zarRank: number;
  zarValue: number;
  adpRank: number;
  adpPick: number;
  rankDelta: number;
  provider: string;
}

export interface ADPReport {
  season: number;
  system: string;
  version: string;
  provider: string;
  buyTargets: ADPReportRow[];
  avoidList: ADPReportRow[];
  unrankedValuable: ADPReportRow[];
  nMatched: number;
}

export interface PlayerSummary {
  playerId: number;
  name: string;
  team: string;
  age: number | null;
  primaryPosition: string;
  bats: string | null;
  throws: string | null;
  experience: number;
}
