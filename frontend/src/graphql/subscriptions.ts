import { gql } from "@apollo/client";

export const DRAFT_EVENTS_SUBSCRIPTION = gql`
  subscription DraftEvents($sessionId: Int!) {
    draftEvents(sessionId: $sessionId) {
      __typename
      ... on PickEvent {
        sessionId
        pick {
          pickNumber
          team
          playerId
          playerName
          position
          price
        }
      }
      ... on UndoEvent {
        sessionId
        pick {
          pickNumber
          team
          playerId
          playerName
          position
          price
        }
      }
      ... on SessionEvent {
        sessionId
        eventType
      }
      ... on ArbitrageAlertEvent {
        sessionId
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
      }
    }
  }
`;
