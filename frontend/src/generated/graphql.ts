import { TypedDocumentNode as DocumentNode } from '@graphql-typed-document-node/core';
export type Maybe<T> = T | null;
export type InputMaybe<T> = Maybe<T>;
export type Exact<T extends { [key: string]: unknown }> = { [K in keyof T]: T[K] };
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]?: Maybe<T[SubKey]> };
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]: Maybe<T[SubKey]> };
export type MakeEmpty<T extends { [key: string]: unknown }, K extends keyof T> = { [_ in K]?: never };
export type Incremental<T> = T | { [P in keyof T]?: P extends ' $fragmentName' | '__typename' ? T[P] : never };
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: { input: string; output: string; }
  String: { input: string; output: string; }
  Boolean: { input: boolean; output: boolean; }
  Int: { input: number; output: number; }
  Float: { input: number; output: number; }
  /** The `JSON` scalar type represents JSON values as specified by [ECMA-404](https://ecma-international.org/wp-content/uploads/ECMA-404_2nd_edition_december_2017.pdf). */
  JSON: { input: Record<string, unknown>; output: Record<string, unknown>; }
};

export type AdpReportRowType = {
  __typename?: 'ADPReportRowType';
  adpPick: Scalars['Float']['output'];
  adpRank: Scalars['Int']['output'];
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  playerType: Scalars['String']['output'];
  position: Position;
  provider: Scalars['String']['output'];
  rankDelta: Scalars['Int']['output'];
  zarRank: Scalars['Int']['output'];
  zarValue: Scalars['Float']['output'];
};

export type AdpReportType = {
  __typename?: 'ADPReportType';
  avoidList: Array<AdpReportRowType>;
  buyTargets: Array<AdpReportRowType>;
  nMatched: Scalars['Int']['output'];
  provider: Scalars['String']['output'];
  season: Scalars['Int']['output'];
  system: Scalars['String']['output'];
  unrankedValuable: Array<AdpReportRowType>;
  version: Scalars['String']['output'];
};

export type AdjustedValuationType = {
  __typename?: 'AdjustedValuationType';
  adjustedValue: Scalars['Float']['output'];
  originalValue: Scalars['Float']['output'];
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  playerType: Scalars['String']['output'];
  position: Position;
  valueChange: Scalars['Float']['output'];
};

export type ArbitrageAlertEvent = {
  __typename?: 'ArbitrageAlertEvent';
  falling: Array<FallingPlayerType>;
  sessionId: Scalars['Int']['output'];
};

export type ArbitrageReportType = {
  __typename?: 'ArbitrageReportType';
  currentPick: Scalars['Int']['output'];
  falling: Array<FallingPlayerType>;
  reaches: Array<ReachPickType>;
};

export type CategoryBalanceType = {
  __typename?: 'CategoryBalanceType';
  category: Scalars['String']['output'];
  leagueRankEstimate: Scalars['Int']['output'];
  projectedValue: Scalars['Float']['output'];
  strength: Scalars['String']['output'];
};

export type CategoryConfigType = {
  __typename?: 'CategoryConfigType';
  direction: Scalars['String']['output'];
  key: Scalars['String']['output'];
  name: Scalars['String']['output'];
  statType: Scalars['String']['output'];
};

export type CategoryNeedType = {
  __typename?: 'CategoryNeedType';
  bestAvailable: Array<PlayerRecommendationType>;
  category: Scalars['String']['output'];
  currentRank: Scalars['Int']['output'];
  targetRank: Scalars['Int']['output'];
};

export type DraftBoardRowType = {
  __typename?: 'DraftBoardRowType';
  adpDelta: Maybe<Scalars['Int']['output']>;
  adpOverall: Maybe<Scalars['Float']['output']>;
  adpRank: Maybe<Scalars['Int']['output']>;
  age: Maybe<Scalars['Int']['output']>;
  batsThrows: Maybe<Scalars['String']['output']>;
  breakoutRank: Maybe<Scalars['Int']['output']>;
  bustRank: Maybe<Scalars['Int']['output']>;
  categoryZScores: Scalars['JSON']['output'];
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  playerType: Scalars['String']['output'];
  position: Position;
  rank: Scalars['Int']['output'];
  tier: Maybe<Scalars['Int']['output']>;
  value: Scalars['Float']['output'];
};

export type DraftBoardType = {
  __typename?: 'DraftBoardType';
  battingCategories: Array<Scalars['String']['output']>;
  pitchingCategories: Array<Scalars['String']['output']>;
  rows: Array<DraftBoardRowType>;
};

export type DraftEventType = ArbitrageAlertEvent | PickEvent | SessionEvent | UndoEvent;

export type DraftPickType = {
  __typename?: 'DraftPickType';
  pickNumber: Scalars['Int']['output'];
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  position: Position;
  price: Maybe<Scalars['Int']['output']>;
  team: Scalars['Int']['output'];
};

export type DraftSessionSummaryType = {
  __typename?: 'DraftSessionSummaryType';
  createdAt: Scalars['String']['output'];
  format: Scalars['String']['output'];
  id: Scalars['Int']['output'];
  league: Scalars['String']['output'];
  pickCount: Scalars['Int']['output'];
  season: Scalars['Int']['output'];
  status: Scalars['String']['output'];
  system: Scalars['String']['output'];
  teams: Scalars['Int']['output'];
  updatedAt: Scalars['String']['output'];
  userTeam: Scalars['Int']['output'];
  version: Scalars['String']['output'];
};

export type DraftStateType = {
  __typename?: 'DraftStateType';
  budgetRemaining: Maybe<Scalars['Int']['output']>;
  currentPick: Scalars['Int']['output'];
  format: Scalars['String']['output'];
  keeperCount: Scalars['Int']['output'];
  picks: Array<DraftPickType>;
  sessionId: Scalars['Int']['output'];
  teams: Scalars['Int']['output'];
  userTeam: Scalars['Int']['output'];
};

export type FallingPlayerType = {
  __typename?: 'FallingPlayerType';
  adp: Scalars['Float']['output'];
  arbitrageScore: Scalars['Float']['output'];
  currentPick: Scalars['Int']['output'];
  picksPastAdp: Scalars['Float']['output'];
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  position: Scalars['String']['output'];
  value: Scalars['Float']['output'];
  valueRank: Scalars['Int']['output'];
};

export type KeeperDecisionType = {
  __typename?: 'KeeperDecisionType';
  cost: Scalars['Float']['output'];
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  position: Position;
  projectedValue: Scalars['Float']['output'];
  recommendation: Scalars['String']['output'];
  surplus: Scalars['Float']['output'];
};

export type KeeperInfoType = {
  __typename?: 'KeeperInfoType';
  cost: Maybe<Scalars['Float']['output']>;
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  position: Scalars['String']['output'];
  teamName: Scalars['String']['output'];
  value: Scalars['Float']['output'];
};

export type KeeperPlanType = {
  __typename?: 'KeeperPlanType';
  scenarios: Array<KeeperScenarioType>;
};

export type KeeperScenarioType = {
  __typename?: 'KeeperScenarioType';
  boardPreview: Array<AdjustedValuationType>;
  categoryNeeds: Array<CategoryNeedType>;
  keeperIds: Array<Scalars['Int']['output']>;
  keepers: Array<KeeperDecisionType>;
  scarcity: Array<PositionScarcityType>;
  strongestCategories: Array<Scalars['String']['output']>;
  totalSurplus: Scalars['Float']['output'];
  weakestCategories: Array<Scalars['String']['output']>;
};

export type LeagueSettingsType = {
  __typename?: 'LeagueSettingsType';
  battingCategories: Array<CategoryConfigType>;
  budget: Scalars['Int']['output'];
  format: Scalars['String']['output'];
  name: Scalars['String']['output'];
  pitcherPositions: Scalars['JSON']['output'];
  pitchingCategories: Array<CategoryConfigType>;
  positions: Scalars['JSON']['output'];
  rosterBatters: Scalars['Int']['output'];
  rosterPitchers: Scalars['Int']['output'];
  rosterUtil: Scalars['Int']['output'];
  teams: Scalars['Int']['output'];
};

export type Mutation = {
  __typename?: 'Mutation';
  endSession: Scalars['Boolean']['output'];
  pick: PickResultType;
  startSession: DraftStateType;
  startYahooPoll: Scalars['Boolean']['output'];
  stopYahooPoll: Scalars['Boolean']['output'];
  undo: PickResultType;
};


export type MutationEndSessionArgs = {
  sessionId: Scalars['Int']['input'];
};


export type MutationPickArgs = {
  playerId: Scalars['Int']['input'];
  position: Position;
  price?: InputMaybe<Scalars['Int']['input']>;
  sessionId: Scalars['Int']['input'];
  team?: InputMaybe<Scalars['Int']['input']>;
};


export type MutationStartSessionArgs = {
  budget?: InputMaybe<Scalars['Int']['input']>;
  format?: Scalars['String']['input'];
  keeperPlayerIds?: InputMaybe<Array<Scalars['Int']['input']>>;
  season: Scalars['Int']['input'];
  system?: InputMaybe<Scalars['String']['input']>;
  teams?: InputMaybe<Scalars['Int']['input']>;
  userTeam?: Scalars['Int']['input'];
  version?: InputMaybe<Scalars['String']['input']>;
};


export type MutationStartYahooPollArgs = {
  leagueKey: Scalars['String']['input'];
  sessionId: Scalars['Int']['input'];
};


export type MutationStopYahooPollArgs = {
  sessionId: Scalars['Int']['input'];
};


export type MutationUndoArgs = {
  sessionId: Scalars['Int']['input'];
};

export type PickEvent = {
  __typename?: 'PickEvent';
  pick: DraftPickType;
  sessionId: Scalars['Int']['output'];
};

export type PickResultType = {
  __typename?: 'PickResultType';
  arbitrage: Maybe<ArbitrageReportType>;
  needs: Array<RosterSlotType>;
  pick: DraftPickType;
  recommendations: Array<RecommendationType>;
  roster: Array<DraftPickType>;
  state: DraftStateType;
};

export type PlayerRecommendationType = {
  __typename?: 'PlayerRecommendationType';
  categoryImpact: Scalars['Float']['output'];
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  tradeoffCategories: Array<Scalars['String']['output']>;
};

export type PlayerSummaryType = {
  __typename?: 'PlayerSummaryType';
  age: Maybe<Scalars['Int']['output']>;
  bats: Maybe<Scalars['String']['output']>;
  experience: Scalars['Int']['output'];
  name: Scalars['String']['output'];
  playerId: Scalars['Int']['output'];
  primaryPosition: Scalars['String']['output'];
  team: Scalars['String']['output'];
  throws: Maybe<Scalars['String']['output']>;
};

export type PlayerTierType = {
  __typename?: 'PlayerTierType';
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  position: Position;
  rank: Scalars['Int']['output'];
  tier: Scalars['Int']['output'];
  value: Scalars['Float']['output'];
};

export type Position =
  | 'BN'
  | 'C'
  | 'CF'
  | 'DH'
  | 'FIRST_BASE'
  | 'LF'
  | 'OF'
  | 'P'
  | 'RF'
  | 'RP'
  | 'SECOND_BASE'
  | 'SP'
  | 'SS'
  | 'THIRD_BASE'
  | 'UTIL';

export type PositionScarcityType = {
  __typename?: 'PositionScarcityType';
  dropoffSlope: Scalars['Float']['output'];
  position: Position;
  replacementValue: Scalars['Float']['output'];
  steepRank: Maybe<Scalars['Int']['output']>;
  tier1Value: Scalars['Float']['output'];
  totalSurplus: Scalars['Float']['output'];
};

export type ProjectionType = {
  __typename?: 'ProjectionType';
  playerId: Maybe<Scalars['Int']['output']>;
  playerName: Scalars['String']['output'];
  playerType: Scalars['String']['output'];
  sourceType: Scalars['String']['output'];
  stats: Scalars['JSON']['output'];
  system: Scalars['String']['output'];
  version: Scalars['String']['output'];
};

export type Query = {
  __typename?: 'Query';
  adpReport: AdpReportType;
  arbitrage: ArbitrageReportType;
  available: Array<DraftBoardRowType>;
  balance: Array<CategoryBalanceType>;
  board: DraftBoardType;
  categoryNeeds: Array<CategoryNeedType>;
  keepers: Array<KeeperInfoType>;
  league: LeagueSettingsType;
  needs: Array<RosterSlotType>;
  planKeeperDraft: KeeperPlanType;
  playerBio: Maybe<PlayerSummaryType>;
  playerSearch: Array<PlayerSummaryType>;
  projectionBoard: Array<ProjectionType>;
  projections: Array<ProjectionType>;
  recommendations: Array<RecommendationType>;
  roster: Array<DraftPickType>;
  scarcity: Array<PositionScarcityType>;
  session: DraftStateType;
  sessions: Array<DraftSessionSummaryType>;
  tiers: Array<PlayerTierType>;
  valuations: Array<ValuationType>;
  webConfig: WebConfigType;
  yahooPollStatus: YahooPollStatusType;
};


export type QueryAdpReportArgs = {
  provider?: InputMaybe<Scalars['String']['input']>;
  season: Scalars['Int']['input'];
  system?: InputMaybe<Scalars['String']['input']>;
  version?: InputMaybe<Scalars['String']['input']>;
};


export type QueryArbitrageArgs = {
  limit?: Scalars['Int']['input'];
  position?: InputMaybe<Position>;
  sessionId: Scalars['Int']['input'];
  threshold?: Scalars['Int']['input'];
};


export type QueryAvailableArgs = {
  limit?: Scalars['Int']['input'];
  position?: InputMaybe<Position>;
  sessionId: Scalars['Int']['input'];
};


export type QueryBalanceArgs = {
  sessionId: Scalars['Int']['input'];
};


export type QueryBoardArgs = {
  playerType?: InputMaybe<Scalars['String']['input']>;
  position?: InputMaybe<Position>;
  season: Scalars['Int']['input'];
  system?: InputMaybe<Scalars['String']['input']>;
  top?: InputMaybe<Scalars['Int']['input']>;
  version?: InputMaybe<Scalars['String']['input']>;
};


export type QueryCategoryNeedsArgs = {
  sessionId: Scalars['Int']['input'];
  topN?: Scalars['Int']['input'];
};


export type QueryKeepersArgs = {
  sessionId: Scalars['Int']['input'];
};


export type QueryNeedsArgs = {
  sessionId: Scalars['Int']['input'];
};


export type QueryPlanKeeperDraftArgs = {
  boardPreviewSize?: Scalars['Int']['input'];
  customScenarios?: InputMaybe<Array<Array<Scalars['Int']['input']>>>;
  maxKeepers: Scalars['Int']['input'];
  season: Scalars['Int']['input'];
  system?: InputMaybe<Scalars['String']['input']>;
  version?: InputMaybe<Scalars['String']['input']>;
};


export type QueryPlayerBioArgs = {
  playerId: Scalars['Int']['input'];
  season: Scalars['Int']['input'];
};


export type QueryPlayerSearchArgs = {
  name: Scalars['String']['input'];
  season: Scalars['Int']['input'];
};


export type QueryProjectionBoardArgs = {
  playerType?: InputMaybe<Scalars['String']['input']>;
  season: Scalars['Int']['input'];
  system: Scalars['String']['input'];
  version: Scalars['String']['input'];
};


export type QueryProjectionsArgs = {
  playerName: Scalars['String']['input'];
  season: Scalars['Int']['input'];
  system?: InputMaybe<Scalars['String']['input']>;
};


export type QueryRecommendationsArgs = {
  limit?: Scalars['Int']['input'];
  position?: InputMaybe<Position>;
  sessionId: Scalars['Int']['input'];
};


export type QueryRosterArgs = {
  sessionId: Scalars['Int']['input'];
  team?: InputMaybe<Scalars['Int']['input']>;
};


export type QueryScarcityArgs = {
  season: Scalars['Int']['input'];
  system?: InputMaybe<Scalars['String']['input']>;
  version?: InputMaybe<Scalars['String']['input']>;
};


export type QuerySessionArgs = {
  sessionId: Scalars['Int']['input'];
};


export type QuerySessionsArgs = {
  league?: InputMaybe<Scalars['String']['input']>;
  season?: InputMaybe<Scalars['Int']['input']>;
  status?: InputMaybe<Scalars['String']['input']>;
};


export type QueryTiersArgs = {
  maxTiers?: Scalars['Int']['input'];
  method?: Scalars['String']['input'];
  playerType?: InputMaybe<Scalars['String']['input']>;
  season: Scalars['Int']['input'];
  system?: InputMaybe<Scalars['String']['input']>;
  version?: InputMaybe<Scalars['String']['input']>;
};


export type QueryValuationsArgs = {
  playerType?: InputMaybe<Scalars['String']['input']>;
  position?: InputMaybe<Scalars['String']['input']>;
  season: Scalars['Int']['input'];
  system?: InputMaybe<Scalars['String']['input']>;
  top?: InputMaybe<Scalars['Int']['input']>;
  version?: InputMaybe<Scalars['String']['input']>;
};


export type QueryYahooPollStatusArgs = {
  sessionId: Scalars['Int']['input'];
};

export type ReachPickType = {
  __typename?: 'ReachPickType';
  adp: Scalars['Float']['output'];
  drafterTeam: Scalars['Int']['output'];
  pickNumber: Scalars['Int']['output'];
  picksAheadOfAdp: Scalars['Float']['output'];
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  position: Scalars['String']['output'];
};

export type RecommendationType = {
  __typename?: 'RecommendationType';
  playerId: Scalars['Int']['output'];
  playerName: Scalars['String']['output'];
  position: Position;
  reason: Scalars['String']['output'];
  score: Scalars['Float']['output'];
  value: Scalars['Float']['output'];
};

export type RosterSlotType = {
  __typename?: 'RosterSlotType';
  position: Position;
  remaining: Scalars['Int']['output'];
};

export type SessionEvent = {
  __typename?: 'SessionEvent';
  eventType: Scalars['String']['output'];
  sessionId: Scalars['Int']['output'];
};

export type Subscription = {
  __typename?: 'Subscription';
  draftEvents: DraftEventType;
};


export type SubscriptionDraftEventsArgs = {
  sessionId: Scalars['Int']['input'];
};

export type SystemVersionType = {
  __typename?: 'SystemVersionType';
  system: Scalars['String']['output'];
  version: Scalars['String']['output'];
};

export type UndoEvent = {
  __typename?: 'UndoEvent';
  pick: DraftPickType;
  sessionId: Scalars['Int']['output'];
};

export type ValuationType = {
  __typename?: 'ValuationType';
  categoryScores: Scalars['JSON']['output'];
  playerName: Scalars['String']['output'];
  playerType: Scalars['String']['output'];
  position: Position;
  projectionSystem: Scalars['String']['output'];
  projectionVersion: Scalars['String']['output'];
  rank: Scalars['Int']['output'];
  system: Scalars['String']['output'];
  value: Scalars['Float']['output'];
  version: Scalars['String']['output'];
};

export type WebConfigType = {
  __typename?: 'WebConfigType';
  projections: Array<SystemVersionType>;
  valuations: Array<SystemVersionType>;
};

export type YahooPollStatusType = {
  __typename?: 'YahooPollStatusType';
  active: Scalars['Boolean']['output'];
  lastPollAt: Maybe<Scalars['String']['output']>;
  picksIngested: Scalars['Int']['output'];
};

export type PickResultFieldsFragment = { __typename?: 'PickResultType', pick: { __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }, state: { __typename?: 'DraftStateType', sessionId: number, currentPick: number, format: string, teams: number, userTeam: number, budgetRemaining: number | null, keeperCount: number, picks: Array<{ __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }> }, recommendations: Array<{ __typename?: 'RecommendationType', playerId: number, playerName: string, position: Position, value: number, score: number, reason: string }>, roster: Array<{ __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }>, needs: Array<{ __typename?: 'RosterSlotType', position: Position, remaining: number }>, arbitrage: { __typename?: 'ArbitrageReportType', currentPick: number, falling: Array<{ __typename?: 'FallingPlayerType', playerId: number, playerName: string, position: string, adp: number, currentPick: number, picksPastAdp: number, value: number, valueRank: number, arbitrageScore: number }>, reaches: Array<{ __typename?: 'ReachPickType', playerId: number, playerName: string, position: string, adp: number, pickNumber: number, picksAheadOfAdp: number, drafterTeam: number }> } | null };

export type StartSessionMutationVariables = Exact<{
  season: Scalars['Int']['input'];
  system: InputMaybe<Scalars['String']['input']>;
  version: InputMaybe<Scalars['String']['input']>;
  teams: InputMaybe<Scalars['Int']['input']>;
  userTeam?: Scalars['Int']['input'];
  format?: Scalars['String']['input'];
  budget: InputMaybe<Scalars['Int']['input']>;
  keeperPlayerIds: InputMaybe<Array<Scalars['Int']['input']> | Scalars['Int']['input']>;
}>;


export type StartSessionMutation = { __typename?: 'Mutation', startSession: { __typename?: 'DraftStateType', sessionId: number, currentPick: number, format: string, teams: number, userTeam: number, budgetRemaining: number | null, keeperCount: number, picks: Array<{ __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }> } };

export type PickMutationVariables = Exact<{
  sessionId: Scalars['Int']['input'];
  playerId: Scalars['Int']['input'];
  position: Position;
  price: InputMaybe<Scalars['Int']['input']>;
  team: InputMaybe<Scalars['Int']['input']>;
}>;


export type PickMutation = { __typename?: 'Mutation', pick: { __typename?: 'PickResultType', pick: { __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }, state: { __typename?: 'DraftStateType', sessionId: number, currentPick: number, format: string, teams: number, userTeam: number, budgetRemaining: number | null, keeperCount: number, picks: Array<{ __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }> }, recommendations: Array<{ __typename?: 'RecommendationType', playerId: number, playerName: string, position: Position, value: number, score: number, reason: string }>, roster: Array<{ __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }>, needs: Array<{ __typename?: 'RosterSlotType', position: Position, remaining: number }>, arbitrage: { __typename?: 'ArbitrageReportType', currentPick: number, falling: Array<{ __typename?: 'FallingPlayerType', playerId: number, playerName: string, position: string, adp: number, currentPick: number, picksPastAdp: number, value: number, valueRank: number, arbitrageScore: number }>, reaches: Array<{ __typename?: 'ReachPickType', playerId: number, playerName: string, position: string, adp: number, pickNumber: number, picksAheadOfAdp: number, drafterTeam: number }> } | null } };

export type UndoMutationVariables = Exact<{
  sessionId: Scalars['Int']['input'];
}>;


export type UndoMutation = { __typename?: 'Mutation', undo: { __typename?: 'PickResultType', pick: { __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }, state: { __typename?: 'DraftStateType', sessionId: number, currentPick: number, format: string, teams: number, userTeam: number, budgetRemaining: number | null, keeperCount: number, picks: Array<{ __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }> }, recommendations: Array<{ __typename?: 'RecommendationType', playerId: number, playerName: string, position: Position, value: number, score: number, reason: string }>, roster: Array<{ __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }>, needs: Array<{ __typename?: 'RosterSlotType', position: Position, remaining: number }>, arbitrage: { __typename?: 'ArbitrageReportType', currentPick: number, falling: Array<{ __typename?: 'FallingPlayerType', playerId: number, playerName: string, position: string, adp: number, currentPick: number, picksPastAdp: number, value: number, valueRank: number, arbitrageScore: number }>, reaches: Array<{ __typename?: 'ReachPickType', playerId: number, playerName: string, position: string, adp: number, pickNumber: number, picksAheadOfAdp: number, drafterTeam: number }> } | null } };

export type EndSessionMutationVariables = Exact<{
  sessionId: Scalars['Int']['input'];
}>;


export type EndSessionMutation = { __typename?: 'Mutation', endSession: boolean };

export type StartYahooPollMutationVariables = Exact<{
  sessionId: Scalars['Int']['input'];
  leagueKey: Scalars['String']['input'];
}>;


export type StartYahooPollMutation = { __typename?: 'Mutation', startYahooPoll: boolean };

export type StopYahooPollMutationVariables = Exact<{
  sessionId: Scalars['Int']['input'];
}>;


export type StopYahooPollMutation = { __typename?: 'Mutation', stopYahooPoll: boolean };

export type BoardQueryVariables = Exact<{
  season: Scalars['Int']['input'];
  system: InputMaybe<Scalars['String']['input']>;
  version: InputMaybe<Scalars['String']['input']>;
}>;


export type BoardQuery = { __typename?: 'Query', board: { __typename?: 'DraftBoardType', battingCategories: Array<string>, pitchingCategories: Array<string>, rows: Array<{ __typename?: 'DraftBoardRowType', playerId: number, playerName: string, rank: number, playerType: string, position: Position, value: number, categoryZScores: Record<string, unknown>, age: number | null, batsThrows: string | null, tier: number | null, adpOverall: number | null, adpRank: number | null, adpDelta: number | null, breakoutRank: number | null, bustRank: number | null }> } };

export type LeagueQueryVariables = Exact<{ [key: string]: never; }>;


export type LeagueQuery = { __typename?: 'Query', league: { __typename?: 'LeagueSettingsType', name: string, format: string, teams: number, budget: number, rosterBatters: number, rosterPitchers: number, rosterUtil: number, battingCategories: Array<{ __typename?: 'CategoryConfigType', key: string, name: string, statType: string, direction: string }>, pitchingCategories: Array<{ __typename?: 'CategoryConfigType', key: string, name: string, statType: string, direction: string }> } };

export type SessionQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
}>;


export type SessionQuery = { __typename?: 'Query', session: { __typename?: 'DraftStateType', sessionId: number, currentPick: number, format: string, teams: number, userTeam: number, budgetRemaining: number | null, keeperCount: number, picks: Array<{ __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }> } };

export type KeepersQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
}>;


export type KeepersQuery = { __typename?: 'Query', keepers: Array<{ __typename?: 'KeeperInfoType', playerId: number, playerName: string, position: string, teamName: string, cost: number | null, value: number }> };

export type SessionsQueryVariables = Exact<{
  league: InputMaybe<Scalars['String']['input']>;
  season: InputMaybe<Scalars['Int']['input']>;
  status: InputMaybe<Scalars['String']['input']>;
}>;


export type SessionsQuery = { __typename?: 'Query', sessions: Array<{ __typename?: 'DraftSessionSummaryType', id: number, league: string, season: number, teams: number, format: string, userTeam: number, status: string, pickCount: number, createdAt: string, updatedAt: string, system: string, version: string }> };

export type RecommendationsQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
  position: InputMaybe<Position>;
  limit: InputMaybe<Scalars['Int']['input']>;
}>;


export type RecommendationsQuery = { __typename?: 'Query', recommendations: Array<{ __typename?: 'RecommendationType', playerId: number, playerName: string, position: Position, value: number, score: number, reason: string }> };

export type RosterQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
  team: InputMaybe<Scalars['Int']['input']>;
}>;


export type RosterQuery = { __typename?: 'Query', roster: Array<{ __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null }> };

export type NeedsQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
}>;


export type NeedsQuery = { __typename?: 'Query', needs: Array<{ __typename?: 'RosterSlotType', position: Position, remaining: number }> };

export type BalanceQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
}>;


export type BalanceQuery = { __typename?: 'Query', balance: Array<{ __typename?: 'CategoryBalanceType', category: string, projectedValue: number, leagueRankEstimate: number, strength: string }> };

export type CategoryNeedsQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
  topN: InputMaybe<Scalars['Int']['input']>;
}>;


export type CategoryNeedsQuery = { __typename?: 'Query', categoryNeeds: Array<{ __typename?: 'CategoryNeedType', category: string, currentRank: number, targetRank: number, bestAvailable: Array<{ __typename?: 'PlayerRecommendationType', playerId: number, playerName: string, categoryImpact: number, tradeoffCategories: Array<string> }> }> };

export type AvailableQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
  position: InputMaybe<Position>;
  limit: InputMaybe<Scalars['Int']['input']>;
}>;


export type AvailableQuery = { __typename?: 'Query', available: Array<{ __typename?: 'DraftBoardRowType', playerId: number, playerName: string, rank: number, playerType: string, position: Position, value: number, categoryZScores: Record<string, unknown>, age: number | null, batsThrows: string | null, tier: number | null, adpOverall: number | null, adpRank: number | null, adpDelta: number | null, breakoutRank: number | null, bustRank: number | null }> };

export type ArbitrageQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
  threshold: InputMaybe<Scalars['Int']['input']>;
  position: InputMaybe<Position>;
  limit: InputMaybe<Scalars['Int']['input']>;
}>;


export type ArbitrageQuery = { __typename?: 'Query', arbitrage: { __typename?: 'ArbitrageReportType', currentPick: number, falling: Array<{ __typename?: 'FallingPlayerType', playerId: number, playerName: string, position: string, adp: number, currentPick: number, picksPastAdp: number, value: number, valueRank: number, arbitrageScore: number }>, reaches: Array<{ __typename?: 'ReachPickType', playerId: number, playerName: string, position: string, adp: number, pickNumber: number, picksAheadOfAdp: number, drafterTeam: number }> } };

export type YahooPollStatusQueryVariables = Exact<{
  sessionId: Scalars['Int']['input'];
}>;


export type YahooPollStatusQuery = { __typename?: 'Query', yahooPollStatus: { __typename?: 'YahooPollStatusType', active: boolean, lastPollAt: string | null, picksIngested: number } };

export type ProjectionsQueryVariables = Exact<{
  season: Scalars['Int']['input'];
  playerName: Scalars['String']['input'];
  system: InputMaybe<Scalars['String']['input']>;
}>;


export type ProjectionsQuery = { __typename?: 'Query', projections: Array<{ __typename?: 'ProjectionType', playerId: number | null, playerName: string, system: string, version: string, sourceType: string, playerType: string, stats: Record<string, unknown> }> };

export type ProjectionBoardQueryVariables = Exact<{
  season: Scalars['Int']['input'];
  system: Scalars['String']['input'];
  version: Scalars['String']['input'];
  playerType: InputMaybe<Scalars['String']['input']>;
}>;


export type ProjectionBoardQuery = { __typename?: 'Query', projectionBoard: Array<{ __typename?: 'ProjectionType', playerId: number | null, playerName: string, system: string, version: string, sourceType: string, playerType: string, stats: Record<string, unknown> }> };

export type WebConfigQueryVariables = Exact<{ [key: string]: never; }>;


export type WebConfigQuery = { __typename?: 'Query', webConfig: { __typename?: 'WebConfigType', projections: Array<{ __typename?: 'SystemVersionType', system: string, version: string }>, valuations: Array<{ __typename?: 'SystemVersionType', system: string, version: string }> } };

export type ValuationsQueryVariables = Exact<{
  season: Scalars['Int']['input'];
  system: InputMaybe<Scalars['String']['input']>;
  version: InputMaybe<Scalars['String']['input']>;
  playerType: InputMaybe<Scalars['String']['input']>;
  position: InputMaybe<Scalars['String']['input']>;
  top: InputMaybe<Scalars['Int']['input']>;
}>;


export type ValuationsQuery = { __typename?: 'Query', valuations: Array<{ __typename?: 'ValuationType', playerName: string, system: string, version: string, projectionSystem: string, projectionVersion: string, playerType: string, position: Position, value: number, rank: number, categoryScores: Record<string, unknown> }> };

export type AdpReportQueryVariables = Exact<{
  season: Scalars['Int']['input'];
  system: InputMaybe<Scalars['String']['input']>;
  version: InputMaybe<Scalars['String']['input']>;
  provider: InputMaybe<Scalars['String']['input']>;
}>;


export type AdpReportQuery = { __typename?: 'Query', adpReport: { __typename?: 'ADPReportType', season: number, system: string, version: string, provider: string, nMatched: number, buyTargets: Array<{ __typename?: 'ADPReportRowType', playerId: number, playerName: string, playerType: string, position: Position, zarRank: number, zarValue: number, adpRank: number, adpPick: number, rankDelta: number, provider: string }>, avoidList: Array<{ __typename?: 'ADPReportRowType', playerId: number, playerName: string, playerType: string, position: Position, zarRank: number, zarValue: number, adpRank: number, adpPick: number, rankDelta: number, provider: string }>, unrankedValuable: Array<{ __typename?: 'ADPReportRowType', playerId: number, playerName: string, playerType: string, position: Position, zarRank: number, zarValue: number, adpRank: number, adpPick: number, rankDelta: number, provider: string }> } };

export type PlayerSearchQueryVariables = Exact<{
  name: Scalars['String']['input'];
  season: Scalars['Int']['input'];
}>;


export type PlayerSearchQuery = { __typename?: 'Query', playerSearch: Array<{ __typename?: 'PlayerSummaryType', playerId: number, name: string, team: string, age: number | null, primaryPosition: string, bats: string | null, throws: string | null, experience: number }> };

export type PlayerBioQueryVariables = Exact<{
  playerId: Scalars['Int']['input'];
  season: Scalars['Int']['input'];
}>;


export type PlayerBioQuery = { __typename?: 'Query', playerBio: { __typename?: 'PlayerSummaryType', playerId: number, name: string, team: string, age: number | null, primaryPosition: string, bats: string | null, throws: string | null, experience: number } | null };

export type PlanKeeperDraftQueryVariables = Exact<{
  season: Scalars['Int']['input'];
  maxKeepers: Scalars['Int']['input'];
  system: InputMaybe<Scalars['String']['input']>;
  version: InputMaybe<Scalars['String']['input']>;
  customScenarios: InputMaybe<Array<Array<Scalars['Int']['input']> | Scalars['Int']['input']> | Array<Scalars['Int']['input']> | Scalars['Int']['input']>;
  boardPreviewSize: InputMaybe<Scalars['Int']['input']>;
}>;


export type PlanKeeperDraftQuery = { __typename?: 'Query', planKeeperDraft: { __typename?: 'KeeperPlanType', scenarios: Array<{ __typename?: 'KeeperScenarioType', keeperIds: Array<number>, totalSurplus: number, strongestCategories: Array<string>, weakestCategories: Array<string>, keepers: Array<{ __typename?: 'KeeperDecisionType', playerId: number, playerName: string, position: Position, cost: number, surplus: number, projectedValue: number, recommendation: string }>, boardPreview: Array<{ __typename?: 'AdjustedValuationType', playerId: number, playerName: string, playerType: string, position: Position, originalValue: number, adjustedValue: number, valueChange: number }>, scarcity: Array<{ __typename?: 'PositionScarcityType', position: Position, tier1Value: number, replacementValue: number, totalSurplus: number, dropoffSlope: number }>, categoryNeeds: Array<{ __typename?: 'CategoryNeedType', category: string, currentRank: number, targetRank: number, bestAvailable: Array<{ __typename?: 'PlayerRecommendationType', playerId: number, playerName: string, categoryImpact: number, tradeoffCategories: Array<string> }> }> }> } };

export type DraftEventsSubscriptionVariables = Exact<{
  sessionId: Scalars['Int']['input'];
}>;


export type DraftEventsSubscription = { __typename?: 'Subscription', draftEvents:
    | { __typename: 'ArbitrageAlertEvent', sessionId: number, falling: Array<{ __typename?: 'FallingPlayerType', playerId: number, playerName: string, position: string, adp: number, currentPick: number, picksPastAdp: number, value: number, valueRank: number, arbitrageScore: number }> }
    | { __typename: 'PickEvent', sessionId: number, pick: { __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null } }
    | { __typename: 'SessionEvent', sessionId: number, eventType: string }
    | { __typename: 'UndoEvent', sessionId: number, pick: { __typename?: 'DraftPickType', pickNumber: number, team: number, playerId: number, playerName: string, position: Position, price: number | null } }
   };

export const PickResultFieldsFragmentDoc = {"kind":"Document","definitions":[{"kind":"FragmentDefinition","name":{"kind":"Name","value":"PickResultFields"},"typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"PickResultType"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pick"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"state"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessionId"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picks"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"format"}},{"kind":"Field","name":{"kind":"Name","value":"teams"}},{"kind":"Field","name":{"kind":"Name","value":"userTeam"}},{"kind":"Field","name":{"kind":"Name","value":"budgetRemaining"}},{"kind":"Field","name":{"kind":"Name","value":"keeperCount"}}]}},{"kind":"Field","name":{"kind":"Name","value":"recommendations"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"reason"}}]}},{"kind":"Field","name":{"kind":"Name","value":"roster"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"needs"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"remaining"}}]}},{"kind":"Field","name":{"kind":"Name","value":"arbitrage"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"falling"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"adp"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picksPastAdp"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"valueRank"}},{"kind":"Field","name":{"kind":"Name","value":"arbitrageScore"}}]}},{"kind":"Field","name":{"kind":"Name","value":"reaches"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"adp"}},{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"picksAheadOfAdp"}},{"kind":"Field","name":{"kind":"Name","value":"drafterTeam"}}]}}]}}]}}]} as unknown as DocumentNode<PickResultFieldsFragment, unknown>;
export const StartSessionDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"mutation","name":{"kind":"Name","value":"StartSession"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"system"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"version"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"teams"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"userTeam"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}},"defaultValue":{"kind":"IntValue","value":"1"}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"format"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},"defaultValue":{"kind":"StringValue","value":"snake","block":false}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"budget"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"keeperPlayerIds"}},"type":{"kind":"ListType","type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"startSession"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}},{"kind":"Argument","name":{"kind":"Name","value":"system"},"value":{"kind":"Variable","name":{"kind":"Name","value":"system"}}},{"kind":"Argument","name":{"kind":"Name","value":"version"},"value":{"kind":"Variable","name":{"kind":"Name","value":"version"}}},{"kind":"Argument","name":{"kind":"Name","value":"teams"},"value":{"kind":"Variable","name":{"kind":"Name","value":"teams"}}},{"kind":"Argument","name":{"kind":"Name","value":"userTeam"},"value":{"kind":"Variable","name":{"kind":"Name","value":"userTeam"}}},{"kind":"Argument","name":{"kind":"Name","value":"format"},"value":{"kind":"Variable","name":{"kind":"Name","value":"format"}}},{"kind":"Argument","name":{"kind":"Name","value":"budget"},"value":{"kind":"Variable","name":{"kind":"Name","value":"budget"}}},{"kind":"Argument","name":{"kind":"Name","value":"keeperPlayerIds"},"value":{"kind":"Variable","name":{"kind":"Name","value":"keeperPlayerIds"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessionId"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picks"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"format"}},{"kind":"Field","name":{"kind":"Name","value":"teams"}},{"kind":"Field","name":{"kind":"Name","value":"userTeam"}},{"kind":"Field","name":{"kind":"Name","value":"budgetRemaining"}},{"kind":"Field","name":{"kind":"Name","value":"keeperCount"}}]}}]}}]} as unknown as DocumentNode<StartSessionMutation, StartSessionMutationVariables>;
export const PickDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"mutation","name":{"kind":"Name","value":"Pick"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"playerId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"position"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Position"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"price"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"team"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pick"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}},{"kind":"Argument","name":{"kind":"Name","value":"playerId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"playerId"}}},{"kind":"Argument","name":{"kind":"Name","value":"position"},"value":{"kind":"Variable","name":{"kind":"Name","value":"position"}}},{"kind":"Argument","name":{"kind":"Name","value":"price"},"value":{"kind":"Variable","name":{"kind":"Name","value":"price"}}},{"kind":"Argument","name":{"kind":"Name","value":"team"},"value":{"kind":"Variable","name":{"kind":"Name","value":"team"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"FragmentSpread","name":{"kind":"Name","value":"PickResultFields"}}]}}]}},{"kind":"FragmentDefinition","name":{"kind":"Name","value":"PickResultFields"},"typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"PickResultType"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pick"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"state"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessionId"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picks"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"format"}},{"kind":"Field","name":{"kind":"Name","value":"teams"}},{"kind":"Field","name":{"kind":"Name","value":"userTeam"}},{"kind":"Field","name":{"kind":"Name","value":"budgetRemaining"}},{"kind":"Field","name":{"kind":"Name","value":"keeperCount"}}]}},{"kind":"Field","name":{"kind":"Name","value":"recommendations"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"reason"}}]}},{"kind":"Field","name":{"kind":"Name","value":"roster"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"needs"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"remaining"}}]}},{"kind":"Field","name":{"kind":"Name","value":"arbitrage"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"falling"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"adp"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picksPastAdp"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"valueRank"}},{"kind":"Field","name":{"kind":"Name","value":"arbitrageScore"}}]}},{"kind":"Field","name":{"kind":"Name","value":"reaches"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"adp"}},{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"picksAheadOfAdp"}},{"kind":"Field","name":{"kind":"Name","value":"drafterTeam"}}]}}]}}]}}]} as unknown as DocumentNode<PickMutation, PickMutationVariables>;
export const UndoDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"mutation","name":{"kind":"Name","value":"Undo"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"undo"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"FragmentSpread","name":{"kind":"Name","value":"PickResultFields"}}]}}]}},{"kind":"FragmentDefinition","name":{"kind":"Name","value":"PickResultFields"},"typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"PickResultType"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pick"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"state"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessionId"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picks"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"format"}},{"kind":"Field","name":{"kind":"Name","value":"teams"}},{"kind":"Field","name":{"kind":"Name","value":"userTeam"}},{"kind":"Field","name":{"kind":"Name","value":"budgetRemaining"}},{"kind":"Field","name":{"kind":"Name","value":"keeperCount"}}]}},{"kind":"Field","name":{"kind":"Name","value":"recommendations"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"reason"}}]}},{"kind":"Field","name":{"kind":"Name","value":"roster"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"needs"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"remaining"}}]}},{"kind":"Field","name":{"kind":"Name","value":"arbitrage"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"falling"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"adp"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picksPastAdp"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"valueRank"}},{"kind":"Field","name":{"kind":"Name","value":"arbitrageScore"}}]}},{"kind":"Field","name":{"kind":"Name","value":"reaches"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"adp"}},{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"picksAheadOfAdp"}},{"kind":"Field","name":{"kind":"Name","value":"drafterTeam"}}]}}]}}]}}]} as unknown as DocumentNode<UndoMutation, UndoMutationVariables>;
export const EndSessionDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"mutation","name":{"kind":"Name","value":"EndSession"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"endSession"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}}]}]}}]} as unknown as DocumentNode<EndSessionMutation, EndSessionMutationVariables>;
export const StartYahooPollDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"mutation","name":{"kind":"Name","value":"StartYahooPoll"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"leagueKey"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"startYahooPoll"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}},{"kind":"Argument","name":{"kind":"Name","value":"leagueKey"},"value":{"kind":"Variable","name":{"kind":"Name","value":"leagueKey"}}}]}]}}]} as unknown as DocumentNode<StartYahooPollMutation, StartYahooPollMutationVariables>;
export const StopYahooPollDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"mutation","name":{"kind":"Name","value":"StopYahooPoll"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"stopYahooPoll"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}}]}]}}]} as unknown as DocumentNode<StopYahooPollMutation, StopYahooPollMutationVariables>;
export const BoardDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Board"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"system"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"version"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"board"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}},{"kind":"Argument","name":{"kind":"Name","value":"system"},"value":{"kind":"Variable","name":{"kind":"Name","value":"system"}}},{"kind":"Argument","name":{"kind":"Name","value":"version"},"value":{"kind":"Variable","name":{"kind":"Name","value":"version"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"rows"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"rank"}},{"kind":"Field","name":{"kind":"Name","value":"playerType"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"categoryZScores"}},{"kind":"Field","name":{"kind":"Name","value":"age"}},{"kind":"Field","name":{"kind":"Name","value":"batsThrows"}},{"kind":"Field","name":{"kind":"Name","value":"tier"}},{"kind":"Field","name":{"kind":"Name","value":"adpOverall"}},{"kind":"Field","name":{"kind":"Name","value":"adpRank"}},{"kind":"Field","name":{"kind":"Name","value":"adpDelta"}},{"kind":"Field","name":{"kind":"Name","value":"breakoutRank"}},{"kind":"Field","name":{"kind":"Name","value":"bustRank"}}]}},{"kind":"Field","name":{"kind":"Name","value":"battingCategories"}},{"kind":"Field","name":{"kind":"Name","value":"pitchingCategories"}}]}}]}}]} as unknown as DocumentNode<BoardQuery, BoardQueryVariables>;
export const LeagueDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"League"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"league"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"name"}},{"kind":"Field","name":{"kind":"Name","value":"format"}},{"kind":"Field","name":{"kind":"Name","value":"teams"}},{"kind":"Field","name":{"kind":"Name","value":"budget"}},{"kind":"Field","name":{"kind":"Name","value":"rosterBatters"}},{"kind":"Field","name":{"kind":"Name","value":"rosterPitchers"}},{"kind":"Field","name":{"kind":"Name","value":"rosterUtil"}},{"kind":"Field","name":{"kind":"Name","value":"battingCategories"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"key"}},{"kind":"Field","name":{"kind":"Name","value":"name"}},{"kind":"Field","name":{"kind":"Name","value":"statType"}},{"kind":"Field","name":{"kind":"Name","value":"direction"}}]}},{"kind":"Field","name":{"kind":"Name","value":"pitchingCategories"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"key"}},{"kind":"Field","name":{"kind":"Name","value":"name"}},{"kind":"Field","name":{"kind":"Name","value":"statType"}},{"kind":"Field","name":{"kind":"Name","value":"direction"}}]}}]}}]}}]} as unknown as DocumentNode<LeagueQuery, LeagueQueryVariables>;
export const SessionDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Session"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"session"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessionId"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picks"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}},{"kind":"Field","name":{"kind":"Name","value":"format"}},{"kind":"Field","name":{"kind":"Name","value":"teams"}},{"kind":"Field","name":{"kind":"Name","value":"userTeam"}},{"kind":"Field","name":{"kind":"Name","value":"budgetRemaining"}},{"kind":"Field","name":{"kind":"Name","value":"keeperCount"}}]}}]}}]} as unknown as DocumentNode<SessionQuery, SessionQueryVariables>;
export const KeepersDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Keepers"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"keepers"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"teamName"}},{"kind":"Field","name":{"kind":"Name","value":"cost"}},{"kind":"Field","name":{"kind":"Name","value":"value"}}]}}]}}]} as unknown as DocumentNode<KeepersQuery, KeepersQueryVariables>;
export const SessionsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Sessions"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"league"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"status"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessions"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"league"},"value":{"kind":"Variable","name":{"kind":"Name","value":"league"}}},{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}},{"kind":"Argument","name":{"kind":"Name","value":"status"},"value":{"kind":"Variable","name":{"kind":"Name","value":"status"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"id"}},{"kind":"Field","name":{"kind":"Name","value":"league"}},{"kind":"Field","name":{"kind":"Name","value":"season"}},{"kind":"Field","name":{"kind":"Name","value":"teams"}},{"kind":"Field","name":{"kind":"Name","value":"format"}},{"kind":"Field","name":{"kind":"Name","value":"userTeam"}},{"kind":"Field","name":{"kind":"Name","value":"status"}},{"kind":"Field","name":{"kind":"Name","value":"pickCount"}},{"kind":"Field","name":{"kind":"Name","value":"createdAt"}},{"kind":"Field","name":{"kind":"Name","value":"updatedAt"}},{"kind":"Field","name":{"kind":"Name","value":"system"}},{"kind":"Field","name":{"kind":"Name","value":"version"}}]}}]}}]} as unknown as DocumentNode<SessionsQuery, SessionsQueryVariables>;
export const RecommendationsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Recommendations"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"position"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Position"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"limit"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"recommendations"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}},{"kind":"Argument","name":{"kind":"Name","value":"position"},"value":{"kind":"Variable","name":{"kind":"Name","value":"position"}}},{"kind":"Argument","name":{"kind":"Name","value":"limit"},"value":{"kind":"Variable","name":{"kind":"Name","value":"limit"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"reason"}}]}}]}}]} as unknown as DocumentNode<RecommendationsQuery, RecommendationsQueryVariables>;
export const RosterDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Roster"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"team"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"roster"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}},{"kind":"Argument","name":{"kind":"Name","value":"team"},"value":{"kind":"Variable","name":{"kind":"Name","value":"team"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}}]}}]} as unknown as DocumentNode<RosterQuery, RosterQueryVariables>;
export const NeedsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Needs"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"needs"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"remaining"}}]}}]}}]} as unknown as DocumentNode<NeedsQuery, NeedsQueryVariables>;
export const BalanceDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Balance"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"balance"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"category"}},{"kind":"Field","name":{"kind":"Name","value":"projectedValue"}},{"kind":"Field","name":{"kind":"Name","value":"leagueRankEstimate"}},{"kind":"Field","name":{"kind":"Name","value":"strength"}}]}}]}}]} as unknown as DocumentNode<BalanceQuery, BalanceQueryVariables>;
export const CategoryNeedsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"CategoryNeeds"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"topN"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"categoryNeeds"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}},{"kind":"Argument","name":{"kind":"Name","value":"topN"},"value":{"kind":"Variable","name":{"kind":"Name","value":"topN"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"category"}},{"kind":"Field","name":{"kind":"Name","value":"currentRank"}},{"kind":"Field","name":{"kind":"Name","value":"targetRank"}},{"kind":"Field","name":{"kind":"Name","value":"bestAvailable"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"categoryImpact"}},{"kind":"Field","name":{"kind":"Name","value":"tradeoffCategories"}}]}}]}}]}}]} as unknown as DocumentNode<CategoryNeedsQuery, CategoryNeedsQueryVariables>;
export const AvailableDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Available"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"position"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Position"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"limit"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"available"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}},{"kind":"Argument","name":{"kind":"Name","value":"position"},"value":{"kind":"Variable","name":{"kind":"Name","value":"position"}}},{"kind":"Argument","name":{"kind":"Name","value":"limit"},"value":{"kind":"Variable","name":{"kind":"Name","value":"limit"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"rank"}},{"kind":"Field","name":{"kind":"Name","value":"playerType"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"categoryZScores"}},{"kind":"Field","name":{"kind":"Name","value":"age"}},{"kind":"Field","name":{"kind":"Name","value":"batsThrows"}},{"kind":"Field","name":{"kind":"Name","value":"tier"}},{"kind":"Field","name":{"kind":"Name","value":"adpOverall"}},{"kind":"Field","name":{"kind":"Name","value":"adpRank"}},{"kind":"Field","name":{"kind":"Name","value":"adpDelta"}},{"kind":"Field","name":{"kind":"Name","value":"breakoutRank"}},{"kind":"Field","name":{"kind":"Name","value":"bustRank"}}]}}]}}]} as unknown as DocumentNode<AvailableQuery, AvailableQueryVariables>;
export const ArbitrageDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Arbitrage"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"threshold"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"position"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Position"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"limit"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"arbitrage"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}},{"kind":"Argument","name":{"kind":"Name","value":"threshold"},"value":{"kind":"Variable","name":{"kind":"Name","value":"threshold"}}},{"kind":"Argument","name":{"kind":"Name","value":"position"},"value":{"kind":"Variable","name":{"kind":"Name","value":"position"}}},{"kind":"Argument","name":{"kind":"Name","value":"limit"},"value":{"kind":"Variable","name":{"kind":"Name","value":"limit"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"falling"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"adp"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picksPastAdp"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"valueRank"}},{"kind":"Field","name":{"kind":"Name","value":"arbitrageScore"}}]}},{"kind":"Field","name":{"kind":"Name","value":"reaches"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"adp"}},{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"picksAheadOfAdp"}},{"kind":"Field","name":{"kind":"Name","value":"drafterTeam"}}]}}]}}]}}]} as unknown as DocumentNode<ArbitrageQuery, ArbitrageQueryVariables>;
export const YahooPollStatusDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"YahooPollStatus"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"yahooPollStatus"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"active"}},{"kind":"Field","name":{"kind":"Name","value":"lastPollAt"}},{"kind":"Field","name":{"kind":"Name","value":"picksIngested"}}]}}]}}]} as unknown as DocumentNode<YahooPollStatusQuery, YahooPollStatusQueryVariables>;
export const ProjectionsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Projections"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"playerName"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"system"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"projections"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}},{"kind":"Argument","name":{"kind":"Name","value":"playerName"},"value":{"kind":"Variable","name":{"kind":"Name","value":"playerName"}}},{"kind":"Argument","name":{"kind":"Name","value":"system"},"value":{"kind":"Variable","name":{"kind":"Name","value":"system"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"system"}},{"kind":"Field","name":{"kind":"Name","value":"version"}},{"kind":"Field","name":{"kind":"Name","value":"sourceType"}},{"kind":"Field","name":{"kind":"Name","value":"playerType"}},{"kind":"Field","name":{"kind":"Name","value":"stats"}}]}}]}}]} as unknown as DocumentNode<ProjectionsQuery, ProjectionsQueryVariables>;
export const ProjectionBoardDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"ProjectionBoard"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"system"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"version"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"playerType"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"projectionBoard"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}},{"kind":"Argument","name":{"kind":"Name","value":"system"},"value":{"kind":"Variable","name":{"kind":"Name","value":"system"}}},{"kind":"Argument","name":{"kind":"Name","value":"version"},"value":{"kind":"Variable","name":{"kind":"Name","value":"version"}}},{"kind":"Argument","name":{"kind":"Name","value":"playerType"},"value":{"kind":"Variable","name":{"kind":"Name","value":"playerType"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"system"}},{"kind":"Field","name":{"kind":"Name","value":"version"}},{"kind":"Field","name":{"kind":"Name","value":"sourceType"}},{"kind":"Field","name":{"kind":"Name","value":"playerType"}},{"kind":"Field","name":{"kind":"Name","value":"stats"}}]}}]}}]} as unknown as DocumentNode<ProjectionBoardQuery, ProjectionBoardQueryVariables>;
export const WebConfigDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"WebConfig"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"webConfig"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"projections"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"system"}},{"kind":"Field","name":{"kind":"Name","value":"version"}}]}},{"kind":"Field","name":{"kind":"Name","value":"valuations"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"system"}},{"kind":"Field","name":{"kind":"Name","value":"version"}}]}}]}}]}}]} as unknown as DocumentNode<WebConfigQuery, WebConfigQueryVariables>;
export const ValuationsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"Valuations"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"system"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"version"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"playerType"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"position"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"top"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"valuations"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}},{"kind":"Argument","name":{"kind":"Name","value":"system"},"value":{"kind":"Variable","name":{"kind":"Name","value":"system"}}},{"kind":"Argument","name":{"kind":"Name","value":"version"},"value":{"kind":"Variable","name":{"kind":"Name","value":"version"}}},{"kind":"Argument","name":{"kind":"Name","value":"playerType"},"value":{"kind":"Variable","name":{"kind":"Name","value":"playerType"}}},{"kind":"Argument","name":{"kind":"Name","value":"position"},"value":{"kind":"Variable","name":{"kind":"Name","value":"position"}}},{"kind":"Argument","name":{"kind":"Name","value":"top"},"value":{"kind":"Variable","name":{"kind":"Name","value":"top"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"system"}},{"kind":"Field","name":{"kind":"Name","value":"version"}},{"kind":"Field","name":{"kind":"Name","value":"projectionSystem"}},{"kind":"Field","name":{"kind":"Name","value":"projectionVersion"}},{"kind":"Field","name":{"kind":"Name","value":"playerType"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"rank"}},{"kind":"Field","name":{"kind":"Name","value":"categoryScores"}}]}}]}}]} as unknown as DocumentNode<ValuationsQuery, ValuationsQueryVariables>;
export const AdpReportDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"ADPReport"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"system"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"version"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"provider"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"adpReport"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}},{"kind":"Argument","name":{"kind":"Name","value":"system"},"value":{"kind":"Variable","name":{"kind":"Name","value":"system"}}},{"kind":"Argument","name":{"kind":"Name","value":"version"},"value":{"kind":"Variable","name":{"kind":"Name","value":"version"}}},{"kind":"Argument","name":{"kind":"Name","value":"provider"},"value":{"kind":"Variable","name":{"kind":"Name","value":"provider"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"season"}},{"kind":"Field","name":{"kind":"Name","value":"system"}},{"kind":"Field","name":{"kind":"Name","value":"version"}},{"kind":"Field","name":{"kind":"Name","value":"provider"}},{"kind":"Field","name":{"kind":"Name","value":"buyTargets"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"playerType"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"zarRank"}},{"kind":"Field","name":{"kind":"Name","value":"zarValue"}},{"kind":"Field","name":{"kind":"Name","value":"adpRank"}},{"kind":"Field","name":{"kind":"Name","value":"adpPick"}},{"kind":"Field","name":{"kind":"Name","value":"rankDelta"}},{"kind":"Field","name":{"kind":"Name","value":"provider"}}]}},{"kind":"Field","name":{"kind":"Name","value":"avoidList"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"playerType"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"zarRank"}},{"kind":"Field","name":{"kind":"Name","value":"zarValue"}},{"kind":"Field","name":{"kind":"Name","value":"adpRank"}},{"kind":"Field","name":{"kind":"Name","value":"adpPick"}},{"kind":"Field","name":{"kind":"Name","value":"rankDelta"}},{"kind":"Field","name":{"kind":"Name","value":"provider"}}]}},{"kind":"Field","name":{"kind":"Name","value":"unrankedValuable"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"playerType"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"zarRank"}},{"kind":"Field","name":{"kind":"Name","value":"zarValue"}},{"kind":"Field","name":{"kind":"Name","value":"adpRank"}},{"kind":"Field","name":{"kind":"Name","value":"adpPick"}},{"kind":"Field","name":{"kind":"Name","value":"rankDelta"}},{"kind":"Field","name":{"kind":"Name","value":"provider"}}]}},{"kind":"Field","name":{"kind":"Name","value":"nMatched"}}]}}]}}]} as unknown as DocumentNode<AdpReportQuery, AdpReportQueryVariables>;
export const PlayerSearchDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"PlayerSearch"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"name"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerSearch"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"name"},"value":{"kind":"Variable","name":{"kind":"Name","value":"name"}}},{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"name"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"age"}},{"kind":"Field","name":{"kind":"Name","value":"primaryPosition"}},{"kind":"Field","name":{"kind":"Name","value":"bats"}},{"kind":"Field","name":{"kind":"Name","value":"throws"}},{"kind":"Field","name":{"kind":"Name","value":"experience"}}]}}]}}]} as unknown as DocumentNode<PlayerSearchQuery, PlayerSearchQueryVariables>;
export const PlayerBioDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"PlayerBio"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"playerId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerBio"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"playerId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"playerId"}}},{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"name"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"age"}},{"kind":"Field","name":{"kind":"Name","value":"primaryPosition"}},{"kind":"Field","name":{"kind":"Name","value":"bats"}},{"kind":"Field","name":{"kind":"Name","value":"throws"}},{"kind":"Field","name":{"kind":"Name","value":"experience"}}]}}]}}]} as unknown as DocumentNode<PlayerBioQuery, PlayerBioQueryVariables>;
export const PlanKeeperDraftDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"PlanKeeperDraft"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"season"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"maxKeepers"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"system"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"version"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"customScenarios"}},"type":{"kind":"ListType","type":{"kind":"NonNullType","type":{"kind":"ListType","type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"boardPreviewSize"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"planKeeperDraft"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"season"},"value":{"kind":"Variable","name":{"kind":"Name","value":"season"}}},{"kind":"Argument","name":{"kind":"Name","value":"maxKeepers"},"value":{"kind":"Variable","name":{"kind":"Name","value":"maxKeepers"}}},{"kind":"Argument","name":{"kind":"Name","value":"system"},"value":{"kind":"Variable","name":{"kind":"Name","value":"system"}}},{"kind":"Argument","name":{"kind":"Name","value":"version"},"value":{"kind":"Variable","name":{"kind":"Name","value":"version"}}},{"kind":"Argument","name":{"kind":"Name","value":"customScenarios"},"value":{"kind":"Variable","name":{"kind":"Name","value":"customScenarios"}}},{"kind":"Argument","name":{"kind":"Name","value":"boardPreviewSize"},"value":{"kind":"Variable","name":{"kind":"Name","value":"boardPreviewSize"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"scenarios"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"keeperIds"}},{"kind":"Field","name":{"kind":"Name","value":"keepers"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"cost"}},{"kind":"Field","name":{"kind":"Name","value":"surplus"}},{"kind":"Field","name":{"kind":"Name","value":"projectedValue"}},{"kind":"Field","name":{"kind":"Name","value":"recommendation"}}]}},{"kind":"Field","name":{"kind":"Name","value":"totalSurplus"}},{"kind":"Field","name":{"kind":"Name","value":"boardPreview"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"playerType"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"originalValue"}},{"kind":"Field","name":{"kind":"Name","value":"adjustedValue"}},{"kind":"Field","name":{"kind":"Name","value":"valueChange"}}]}},{"kind":"Field","name":{"kind":"Name","value":"scarcity"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"tier1Value"}},{"kind":"Field","name":{"kind":"Name","value":"replacementValue"}},{"kind":"Field","name":{"kind":"Name","value":"totalSurplus"}},{"kind":"Field","name":{"kind":"Name","value":"dropoffSlope"}}]}},{"kind":"Field","name":{"kind":"Name","value":"categoryNeeds"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"category"}},{"kind":"Field","name":{"kind":"Name","value":"currentRank"}},{"kind":"Field","name":{"kind":"Name","value":"targetRank"}},{"kind":"Field","name":{"kind":"Name","value":"bestAvailable"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"categoryImpact"}},{"kind":"Field","name":{"kind":"Name","value":"tradeoffCategories"}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"strongestCategories"}},{"kind":"Field","name":{"kind":"Name","value":"weakestCategories"}}]}}]}}]}}]} as unknown as DocumentNode<PlanKeeperDraftQuery, PlanKeeperDraftQueryVariables>;
export const DraftEventsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"subscription","name":{"kind":"Name","value":"DraftEvents"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"draftEvents"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"sessionId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sessionId"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"__typename"}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"PickEvent"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessionId"}},{"kind":"Field","name":{"kind":"Name","value":"pick"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}}]}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"UndoEvent"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessionId"}},{"kind":"Field","name":{"kind":"Name","value":"pick"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"pickNumber"}},{"kind":"Field","name":{"kind":"Name","value":"team"}},{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"price"}}]}}]}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"SessionEvent"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessionId"}},{"kind":"Field","name":{"kind":"Name","value":"eventType"}}]}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"ArbitrageAlertEvent"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"sessionId"}},{"kind":"Field","name":{"kind":"Name","value":"falling"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"playerId"}},{"kind":"Field","name":{"kind":"Name","value":"playerName"}},{"kind":"Field","name":{"kind":"Name","value":"position"}},{"kind":"Field","name":{"kind":"Name","value":"adp"}},{"kind":"Field","name":{"kind":"Name","value":"currentPick"}},{"kind":"Field","name":{"kind":"Name","value":"picksPastAdp"}},{"kind":"Field","name":{"kind":"Name","value":"value"}},{"kind":"Field","name":{"kind":"Name","value":"valueRank"}},{"kind":"Field","name":{"kind":"Name","value":"arbitrageScore"}}]}}]}}]}}]}}]} as unknown as DocumentNode<DraftEventsSubscription, DraftEventsSubscriptionVariables>;