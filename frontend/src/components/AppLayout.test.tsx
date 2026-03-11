import type { MockedResponse } from "@apollo/client/testing";
import { MockedProvider } from "@apollo/client/testing";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import { WEB_CONFIG_QUERY } from "../graphql/queries";
import { AppLayout } from "./AppLayout";

const NO_YAHOO_MOCK: MockedResponse = {
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

const WITH_YAHOO_MOCK: MockedResponse = {
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

function renderLayout(mocks: MockedResponse[] = [NO_YAHOO_MOCK], initialEntry = "/") {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <PlayerDrawerProvider>
        <MemoryRouter initialEntries={[initialEntry]}>
          <AppLayout />
        </MemoryRouter>
      </PlayerDrawerProvider>
    </MockedProvider>,
  );
}

describe("AppLayout", () => {
  afterEach(cleanup);

  it("renders navigation links", () => {
    renderLayout();
    expect(screen.getByText("Draft")).toBeInTheDocument();
    expect(screen.getByText("Projections")).toBeInTheDocument();
    expect(screen.getByText("Valuations")).toBeInTheDocument();
    expect(screen.getByText("ADP Report")).toBeInTheDocument();
    expect(screen.getByText("Player Search")).toBeInTheDocument();
  });

  it("renders FBM brand", () => {
    renderLayout();
    expect(screen.getByText("FBM")).toBeInTheDocument();
  });

  it("highlights active link", () => {
    renderLayout([NO_YAHOO_MOCK], "/projections");
    const link = screen.getByText("Projections");
    expect(link.className).toContain("bg-blue-600");
  });

  it("shows league badge when yahoo league is configured", async () => {
    renderLayout([WITH_YAHOO_MOCK]);
    await waitFor(() => {
      expect(screen.getByText("Dynasty Kings")).toBeInTheDocument();
    });
    expect(screen.getByText("2026")).toBeInTheDocument();
  });

  it("hides league badge when yahoo league is not configured", async () => {
    renderLayout([NO_YAHOO_MOCK]);
    // Wait for query to resolve
    await waitFor(() => {
      expect(screen.getByText("FBM")).toBeInTheDocument();
    });
    expect(screen.queryByText("Dynasty Kings")).not.toBeInTheDocument();
  });
});
