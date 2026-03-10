import { gql } from "@apollo/client";

export {
  AdpReportDocument as ADP_REPORT_QUERY,
  ArbitrageDocument as ARBITRAGE_QUERY,
  AvailableDocument as AVAILABLE_QUERY,
  BalanceDocument as BALANCE_QUERY,
  BoardDocument as BOARD_QUERY,
  LeagueDocument as LEAGUE_QUERY,
  NeedsDocument as NEEDS_QUERY,
  PlayerBioDocument as PLAYER_BIO_QUERY,
  PlayerSearchDocument as PLAYER_SEARCH_QUERY,
  ProjectionsDocument as PROJECTIONS_QUERY,
  RecommendationsDocument as RECOMMENDATIONS_QUERY,
  RosterDocument as ROSTER_QUERY,
  SessionDocument as SESSION_QUERY,
  SessionsDocument as SESSIONS_QUERY,
  ValuationsDocument as VALUATIONS_QUERY,
  YahooPollStatusDocument as YAHOO_POLL_STATUS_QUERY,
} from "../generated/graphql";

// KEEPERS_QUERY is not yet in the codegen schema — keep as hand-written gql until
// the keeper fields are added to the codegen configuration.
export const KEEPERS_QUERY = gql`
  query Keepers($sessionId: Int!) {
    keepers(sessionId: $sessionId) {
      playerId
      playerName
      position
      teamName
      cost
      value
    }
  }
`;
