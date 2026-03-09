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
