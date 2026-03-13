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
      teamNames
      draftOrder
      trades {
        teamA
        teamB
        teamAGives
        teamBGives
      }
    }
  }
`;

export const KEEPERS_QUERY = gql`
  query Keepers($sessionId: Int!) {
    keepers(sessionId: $sessionId) {
      playerId
      playerName
      playerType
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

export const CATEGORY_NEEDS_QUERY = gql`
  query CategoryNeeds($sessionId: Int!, $topN: Int) {
    categoryNeeds(sessionId: $sessionId, topN: $topN) {
      category
      currentRank
      targetRank
      bestAvailable {
        playerId
        playerName
        categoryImpact
        tradeoffCategories
      }
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
  query Arbitrage($sessionId: Int!, $threshold: Int, $position: Position, $limit: Int) {
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
      playerId
      playerName
      system
      version
      sourceType
      playerType
      stats
    }
  }
`;

export const PROJECTION_BOARD_QUERY = gql`
  query ProjectionBoard($season: Int!, $system: String!, $version: String!, $playerType: String) {
    projectionBoard(season: $season, system: $system, version: $version, playerType: $playerType) {
      playerId
      playerName
      system
      version
      sourceType
      playerType
      stats
    }
  }
`;

export const WEB_CONFIG_QUERY = gql`
  query WebConfig {
    webConfig {
      projections {
        system
        version
      }
      valuations {
        system
        version
      }
      yahooLeague {
        leagueKey
        leagueName
        season
        numTeams
        isKeeper
        maxKeepers
        userTeamName
      }
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

export const YAHOO_TEAMS_QUERY = gql`
  query YahooTeams($leagueKey: String!) {
    yahooTeams(leagueKey: $leagueKey) {
      teamKey
      name
      managerName
      isOwnedByUser
    }
  }
`;

export const YAHOO_STANDINGS_QUERY = gql`
  query YahooStandings($leagueKey: String!, $season: Int!) {
    yahooStandings(leagueKey: $leagueKey, season: $season) {
      teamKey
      teamName
      finalRank
      statValues
    }
  }
`;

export const YAHOO_ROSTERS_QUERY = gql`
  query YahooRosters($leagueKey: String!) {
    yahooRosters(leagueKey: $leagueKey) {
      teamKey
      season
      week
      asOf
      entries {
        yahooPlayerKey
        playerName
        position
        acquisitionType
        playerId
      }
    }
  }
`;

export const YAHOO_KEEPER_OVERVIEW_QUERY = gql`
  query YahooKeeperOverview($leagueKey: String!, $season: Int!, $maxKeepers: Int!) {
    yahooKeeperOverview(leagueKey: $leagueKey, season: $season, maxKeepers: $maxKeepers) {
      teamProjections {
        teamKey
        teamName
        isUser
        totalValue
        categoryTotals
        keepers {
          playerId
          playerName
          position
          value
          categoryScores
        }
      }
      tradeTargets {
        playerId
        playerName
        position
        value
        owningTeamName
        owningTeamKey
        rankOnTeam
      }
      categoryNames
    }
  }
`;

export const EVALUATE_TRADE_QUERY = gql`
  query EvaluateTrade($sessionId: Int!, $gives: [Int!]!, $receives: [Int!]!) {
    evaluateTrade(sessionId: $sessionId, gives: $gives, receives: $receives) {
      givesValue
      receivesValue
      netValue
      givesDetail {
        pickNumber
        value
      }
      receivesDetail {
        pickNumber
        value
      }
      recommendation
    }
  }
`;

export const YAHOO_DRAFT_SETUP_QUERY = gql`
  query YahooDraftSetup($leagueKey: String!, $season: Int!) {
    yahooDraftSetup(leagueKey: $leagueKey, season: $season) {
      numTeams
      draftFormat
      userTeamId
      teamNames
      draftOrder
      isKeeper
      maxKeepers
      keeperPlayerIds
    }
  }
`;

export const PLAN_KEEPER_DRAFT_QUERY = gql`
  query PlanKeeperDraft(
    $season: Int!
    $maxKeepers: Int!
    $system: String
    $version: String
    $customScenarios: [[Int!]!]
    $boardPreviewSize: Int
  ) {
    planKeeperDraft(
      season: $season
      maxKeepers: $maxKeepers
      system: $system
      version: $version
      customScenarios: $customScenarios
      boardPreviewSize: $boardPreviewSize
    ) {
      scenarios {
        keeperIds
        keepers {
          playerId
          playerName
          position
          cost
          surplus
          projectedValue
          recommendation
        }
        totalSurplus
        boardPreview {
          playerId
          playerName
          playerType
          position
          originalValue
          adjustedValue
          valueChange
        }
        scarcity {
          position
          tier1Value
          replacementValue
          totalSurplus
          dropoffSlope
        }
        categoryNeeds {
          category
          currentRank
          targetRank
          bestAvailable {
            playerId
            playerName
            categoryImpact
            tradeoffCategories
          }
        }
        strongestCategories
        weakestCategories
      }
    }
  }
`;
