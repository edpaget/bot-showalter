import type { MockedResponse } from "@apollo/client/testing";
import { MockedProvider } from "@apollo/client/testing";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import { WEB_CONFIG_QUERY, YAHOO_STANDINGS_QUERY, YAHOO_TEAMS_QUERY } from "../graphql/queries";
import { LeagueView } from "./LeagueView";

const CONFIG_WITH_YAHOO: MockedResponse = {
  request: { query: WEB_CONFIG_QUERY },
  result: {
    data: {
      webConfig: {
        projections: [],
        valuations: [],
        yahooLeague: {
          leagueKey: "449.l.12345",
          leagueName: "Dynasty Kings",
          season: 2026,
          numTeams: 12,
          isKeeper: true,
          maxKeepers: 5,
          userTeamName: "My Team",
        },
      },
    },
  },
};

const CONFIG_WITHOUT_YAHOO: MockedResponse = {
  request: { query: WEB_CONFIG_QUERY },
  result: {
    data: {
      webConfig: {
        projections: [],
        valuations: [],
        yahooLeague: null,
      },
    },
  },
};

const TEAMS_MOCK: MockedResponse = {
  request: { query: YAHOO_TEAMS_QUERY, variables: { leagueKey: "449.l.12345" } },
  result: {
    data: {
      yahooTeams: [
        { teamKey: "449.l.12345.t.1", name: "Dynasty Kings", managerName: "Alice", isOwnedByUser: true },
        { teamKey: "449.l.12345.t.2", name: "Rival Squad", managerName: "Bob", isOwnedByUser: false },
      ],
    },
  },
};

const STANDINGS_MOCK: MockedResponse = {
  request: { query: YAHOO_STANDINGS_QUERY, variables: { leagueKey: "449.l.12345", season: 2026 } },
  result: {
    data: {
      yahooStandings: [
        { teamKey: "449.l.12345.t.1", teamName: "Dynasty Kings", finalRank: 1, statValues: { HR: 250, RBI: 800 } },
        { teamKey: "449.l.12345.t.2", teamName: "Rival Squad", finalRank: 2, statValues: { HR: 220, RBI: 750 } },
      ],
    },
  },
};

function renderView(mocks: MockedResponse[]) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <MemoryRouter>
        <LeagueView />
      </MemoryRouter>
    </MockedProvider>,
  );
}

describe("LeagueView", () => {
  afterEach(cleanup);

  it("renders standings table with data", async () => {
    renderView([CONFIG_WITH_YAHOO, TEAMS_MOCK, STANDINGS_MOCK]);

    await waitFor(() => {
      expect(screen.getByText("Dynasty Kings")).toBeInTheDocument();
    });

    expect(screen.getByText("Rival Squad")).toBeInTheDocument();
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Bob")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("shows not-configured message when no yahoo league", async () => {
    renderView([CONFIG_WITHOUT_YAHOO]);

    await waitFor(() => {
      expect(screen.getByText(/No Yahoo league configured/)).toBeInTheDocument();
    });
  });
});
