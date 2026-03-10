import { gql } from "@apollo/client";

export const BOARD_QUERY = gql`
  query Board($season: Int!, $system: String, $version: String) {
    board(season: $season, system: $system, version: $version) {
      rows {
        playerId
        playerName
        rank
        playerType
        position
        value
        categoryZScores
        age
        batsThrows
        tier
        adpOverall
        adpRank
        adpDelta
        breakoutRank
        bustRank
      }
      battingCategories
      pitchingCategories
    }
  }
`;

export const LEAGUE_QUERY = gql`
  query League {
    league {
      name
      format
      teams
      budget
      rosterBatters
      rosterPitchers
      rosterUtil
      battingCategories {
        key
        name
        statType
        direction
      }
      pitchingCategories {
        key
        name
        statType
        direction
      }
    }
  }
`;

export const SESSION_QUERY = gql`
  query Session($sessionId: Int!) {
    session(sessionId: $sessionId) {
      sessionId
      currentPick
      picks {
        pickNumber
        team
        playerId
        playerName
        position
        price
      }
      format
      teams
      userTeam
      budgetRemaining
    }
  }
`;

export const SESSIONS_QUERY = gql`
  query Sessions($league: String, $season: Int, $status: String) {
    sessions(league: $league, season: $season, status: $status) {
      id
      league
      season
      teams
      format
      userTeam
      status
      pickCount
      createdAt
      updatedAt
      system
      version
    }
  }
`;

export const RECOMMENDATIONS_QUERY = gql`
  query Recommendations($sessionId: Int!, $position: Position, $limit: Int) {
    recommendations(sessionId: $sessionId, position: $position, limit: $limit) {
      playerId
      playerName
      position
      value
      score
      reason
    }
  }
`;

export const ROSTER_QUERY = gql`
  query Roster($sessionId: Int!, $team: Int) {
    roster(sessionId: $sessionId, team: $team) {
      pickNumber
      team
      playerId
      playerName
      position
      price
    }
  }
`;

export const NEEDS_QUERY = gql`
  query Needs($sessionId: Int!) {
    needs(sessionId: $sessionId) {
      position
      remaining
    }
  }
`;

export const BALANCE_QUERY = gql`
  query Balance($sessionId: Int!) {
    balance(sessionId: $sessionId) {
      category
      projectedValue
      leagueRankEstimate
      strength
    }
  }
`;

export const AVAILABLE_QUERY = gql`
  query Available($sessionId: Int!, $position: Position, $limit: Int) {
    available(sessionId: $sessionId, position: $position, limit: $limit) {
      playerId
      playerName
      rank
      playerType
      position
      value
      categoryZScores
      age
      batsThrows
      tier
      adpOverall
      adpRank
      adpDelta
      breakoutRank
      bustRank
    }
  }
`;

export const ARBITRAGE_QUERY = gql`
  query Arbitrage($sessionId: Int!, $threshold: Int, $position: String, $limit: Int) {
    arbitrage(sessionId: $sessionId, threshold: $threshold, position: $position, limit: $limit) {
      currentPick
      falling {
        playerId
        playerName
        position
        adp
        currentPick
        picksPastAdp
        value
        valueRank
        arbitrageScore
      }
      reaches {
        playerId
        playerName
        position
        adp
        pickNumber
        picksAheadOfAdp
        drafterTeam
      }
    }
  }
`;

export const YAHOO_POLL_STATUS_QUERY = gql`
  query YahooPollStatus($sessionId: Int!) {
    yahooPollStatus(sessionId: $sessionId) {
      active
      lastPollAt
      picksIngested
    }
  }
`;
