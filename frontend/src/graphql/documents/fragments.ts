import { gql } from "@apollo/client";

export const PICK_RESULT_FRAGMENT = gql`
  fragment PickResultFields on PickResultType {
    pick {
      pickNumber
      team
      playerId
      playerName
      position
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
        price
      }
      format
      teams
      userTeam
      budgetRemaining
      keeperCount
      teamNames
    }
    recommendations {
      playerId
      playerName
      position
      value
      score
      reason
    }
    roster {
      pickNumber
      team
      playerId
      playerName
      position
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
