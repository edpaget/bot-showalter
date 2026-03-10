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
      keeperCount
    }
  }
`;

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

export const PROJECTIONS_QUERY = gql`
  query Projections($season: Int!, $playerName: String!, $system: String) {
    projections(season: $season, playerName: $playerName, system: $system) {
      playerName
      system
      version
      sourceType
      playerType
      stats
    }
  }
`;

export const VALUATIONS_QUERY = gql`
  query Valuations(
    $season: Int!
    $system: String
    $version: String
    $playerType: String
    $position: String
    $top: Int
  ) {
    valuations(
      season: $season
      system: $system
      version: $version
      playerType: $playerType
      position: $position
      top: $top
    ) {
      playerName
      system
      version
      projectionSystem
      projectionVersion
      playerType
      position
      value
      rank
      categoryScores
    }
  }
`;

export const ADP_REPORT_QUERY = gql`
  query ADPReport($season: Int!, $system: String, $version: String, $provider: String) {
    adpReport(season: $season, system: $system, version: $version, provider: $provider) {
      season
      system
      version
      provider
      buyTargets {
        playerId
        playerName
        playerType
        position
        zarRank
        zarValue
        adpRank
        adpPick
        rankDelta
        provider
      }
      avoidList {
        playerId
        playerName
        playerType
        position
        zarRank
        zarValue
        adpRank
        adpPick
        rankDelta
        provider
      }
      unrankedValuable {
        playerId
        playerName
        playerType
        position
        zarRank
        zarValue
        adpRank
        adpPick
        rankDelta
        provider
      }
      nMatched
    }
  }
`;

export const PLAYER_SEARCH_QUERY = gql`
  query PlayerSearch($name: String!, $season: Int!) {
    playerSearch(name: $name, season: $season) {
      playerId
      name
      team
      age
      primaryPosition
      bats
      throws
      experience
    }
  }
`;

export const PLAYER_BIO_QUERY = gql`
  query PlayerBio($playerId: Int!, $season: Int!) {
    playerBio(playerId: $playerId, season: $season) {
      playerId
      name
      team
      age
      primaryPosition
      bats
      throws
      experience
    }
  }
`;
