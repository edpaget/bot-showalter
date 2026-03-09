export interface DraftBoardRow {
  playerId: number;
  playerName: string;
  rank: number;
  playerType: string;
  position: string;
  value: number;
  categoryZScores: Record<string, number>;
  age: number | null;
  batsThrows: string | null;
  tier: number | null;
  adpOverall: number | null;
  adpRank: number | null;
  adpDelta: number | null;
  breakoutRank: number | null;
  bustRank: number | null;
}

export interface DraftBoard {
  rows: DraftBoardRow[];
  battingCategories: string[];
  pitchingCategories: string[];
}

export interface CategoryConfig {
  key: string;
  name: string;
  statType: string;
  direction: string;
}

export interface LeagueSettings {
  name: string;
  format: string;
  teams: number;
  budget: number;
  rosterBatters: number;
  rosterPitchers: number;
  rosterUtil: number;
  battingCategories: CategoryConfig[];
  pitchingCategories: CategoryConfig[];
}
