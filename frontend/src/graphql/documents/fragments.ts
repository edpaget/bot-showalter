import { gql } from "@apollo/client";

export const PICK_RESULT_FRAGMENT = gql`
  fragment PickResultFields on PickResultType {
    pick {
      pickNumber
      team
      playerId
      playerName
      position
      playerType
      price
    }
    state {
      sessionId
      currentPick
      picks {
        pickNumber
        team
        playerId
        playerName
        position
        playerType
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
    recommendations {
      playerId
      playerName
      position
      value
      score
      reason
      playerType
    }
    roster {
      pickNumber
      team
      playerId
      playerName
      position
      playerType
      price
    }
    needs {
      position
      remaining
    }
    arbitrage {
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
        playerType
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
    balance {
      category
      projectedValue
      leagueRankEstimate
      strength
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
  }
`;
