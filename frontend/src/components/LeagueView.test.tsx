import type { MockedResponse } from "@apollo/client/testing";
import { MockedProvider } from "@apollo/client/testing";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import { WEB_CONFIG_QUERY, YAHOO_ROSTERS_QUERY, YAHOO_STANDINGS_QUERY, YAHOO_TEAMS_QUERY } from "../graphql/queries";
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

const ROSTERS_MOCK: MockedResponse = {
  request: { query: YAHOO_ROSTERS_QUERY, variables: { leagueKey: "449.l.12345" } },
  result: {
    data: {
      yahooRosters: [
        {
          teamKey: "449.l.12345.t.1",
          season: 2026,
          week: 1,
          asOf: "2026-03-28",
          entries: [
            {
              yahooPlayerKey: "449.p.10001",
              playerName: "Mike Trout",
              position: "OF",
              acquisitionType: "draft",
              playerId: 1,
            },
            {
              yahooPlayerKey: "449.p.10099",
              playerName: "Unknown Prospect",
              position: "BN",
              acquisitionType: "add",
              playerId: null,
            },
          ],
        },
        {
          teamKey: "449.l.12345.t.2",
          season: 2026,
          week: 1,
          asOf: "2026-03-28",
          entries: [
            {
              yahooPlayerKey: "449.p.10002",
              playerName: "Shohei Ohtani",
              position: "OF",
              acquisitionType: "draft",
              playerId: 2,
            },
          ],
        },
      ],
    },
  },
};

function renderView(mocks: MockedResponse[]) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <MemoryRouter>
        <PlayerDrawerProvider>
          <LeagueView />
        </PlayerDrawerProvider>
      </MemoryRouter>
    </MockedProvider>,
  );
}

describe("LeagueView", () => {
  afterEach(cleanup);

  it("renders standings table with data", async () => {
    renderView([CONFIG_WITH_YAHOO, TEAMS_MOCK, STANDINGS_MOCK, ROSTERS_MOCK]);

    await waitFor(() => {
      expect(screen.getByText("Alice")).toBeInTheDocument();
    });

    expect(screen.getByText("Rival Squad")).toBeInTheDocument();
    expect(screen.getByText("Bob")).toBeInTheDocument();
  });

  it("shows not-configured message when no yahoo league", async () => {
    renderView([CONFIG_WITHOUT_YAHOO]);

    await waitFor(() => {
      expect(screen.getByText(/No Yahoo league configured/)).toBeInTheDocument();
    });
  });

  it("shows roster entries when a team row is clicked", async () => {
    renderView([CONFIG_WITH_YAHOO, TEAMS_MOCK, STANDINGS_MOCK, ROSTERS_MOCK]);

    await waitFor(() => {
      expect(screen.getByText("Alice")).toBeInTheDocument();
    });

    // Roster entries should not be visible initially
    expect(screen.queryByText("Mike Trout")).not.toBeInTheDocument();

    // Click on Dynasty Kings row to expand roster
    const dynastyRow = screen.getByText("Alice").closest("tr") as HTMLElement;
    fireEvent.click(dynastyRow);

    await waitFor(() => {
      expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    });
    expect(screen.getByText("Unknown Prospect")).toBeInTheDocument();
  });

  it("player names with playerId are clickable buttons", async () => {
    renderView([CONFIG_WITH_YAHOO, TEAMS_MOCK, STANDINGS_MOCK, ROSTERS_MOCK]);

    await waitFor(() => {
      expect(screen.getByText("Alice")).toBeInTheDocument();
    });

    const dynastyRow = screen.getByText("Alice").closest("tr") as HTMLElement;
    fireEvent.click(dynastyRow);

    await waitFor(() => {
      expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    });

    // Mapped player should be a button
    const troutEl = screen.getByText("Mike Trout");
    expect(troutEl.tagName).toBe("BUTTON");

    // Unmapped player should be a span (plain text)
    const prospectEl = screen.getByText("Unknown Prospect");
    expect(prospectEl.tagName).toBe("SPAN");
  });

  it("hides roster when team is deselected", async () => {
    renderView([CONFIG_WITH_YAHOO, TEAMS_MOCK, STANDINGS_MOCK, ROSTERS_MOCK]);

    await waitFor(() => {
      expect(screen.getByText("Alice")).toBeInTheDocument();
    });

    const dynastyRow = screen.getByText("Alice").closest("tr") as HTMLElement;

    // Click to expand
    fireEvent.click(dynastyRow);
    await waitFor(() => {
      expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    });

    // Click again to collapse
    fireEvent.click(dynastyRow);
    await waitFor(() => {
      expect(screen.queryByText("Mike Trout")).not.toBeInTheDocument();
    });
  });
});
