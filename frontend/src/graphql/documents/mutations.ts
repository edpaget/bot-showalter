import { gql } from "@apollo/client";
import { PICK_RESULT_FRAGMENT } from "./fragments";

export const START_SESSION = gql`
  mutation StartSession(
    $season: Int!
    $system: String
    $version: String
    $teams: Int
    $userTeam: Int! = 1
    $format: String! = "snake"
    $budget: Int
  ) {
    startSession(
      season: $season
      system: $system
      version: $version
      teams: $teams
      userTeam: $userTeam
      format: $format
      budget: $budget
    ) {
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

export const PICK = gql`
  mutation Pick(
    $sessionId: Int!
    $playerId: Int!
    $position: Position!
    $price: Int
    $team: Int
  ) {
    pick(
      sessionId: $sessionId
      playerId: $playerId
      position: $position
      price: $price
      team: $team
    ) {
      ...PickResultFields
    }
  }
  ${PICK_RESULT_FRAGMENT}
`;

export const UNDO = gql`
  mutation Undo($sessionId: Int!) {
    undo(sessionId: $sessionId) {
      ...PickResultFields
    }
  }
  ${PICK_RESULT_FRAGMENT}
`;

export const END_SESSION = gql`
  mutation EndSession($sessionId: Int!) {
    endSession(sessionId: $sessionId)
  }
`;

export const START_YAHOO_POLL = gql`
  mutation StartYahooPoll($sessionId: Int!, $leagueKey: String!) {
    startYahooPoll(sessionId: $sessionId, leagueKey: $leagueKey)
  }
`;

export const STOP_YAHOO_POLL = gql`
  mutation StopYahooPoll($sessionId: Int!) {
    stopYahooPoll(sessionId: $sessionId)
  }
`;
