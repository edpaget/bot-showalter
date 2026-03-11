import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import { LEAGUE_QUERY, PROJECTION_BOARD_QUERY, WEB_CONFIG_QUERY } from "../graphql/queries";
import { ProjectionsView } from "./ProjectionsView";

function webConfigMock(): MockedResponse {
  return {
    request: { query: WEB_CONFIG_QUERY },
    result: {
      data: {
        webConfig: {
          projections: [{ system: "steamer", version: "2026" }],
          valuations: [{ system: "steamer", version: "2026" }],
        },
      },
    },
  };
}

function leagueMock(): MockedResponse {
  return {
    request: { query: LEAGUE_QUERY },
    result: {
      data: {
        league: {
          name: "Test League",
          format: "roto",
          teams: 12,
          budget: 260,
          rosterBatters: 14,
          rosterPitchers: 9,
          rosterUtil: 1,
          battingCategories: [
            { key: "hr", name: "HR", statType: "counting", direction: "higher" },
            { key: "rbi", name: "RBI", statType: "counting", direction: "higher" },
            { key: "avg", name: "AVG", statType: "rate", direction: "higher" },
          ],
          pitchingCategories: [
            { key: "w", name: "W", statType: "counting", direction: "higher" },
            { key: "era", name: "ERA", statType: "rate", direction: "lower" },
          ],
          positions: {},
          pitcherPositions: {},
        },
      },
    },
  };
}

function boardMock(): MockedResponse {
  return {
    request: {
      query: PROJECTION_BOARD_QUERY,
      variables: { season: 2026, system: "steamer", version: "2026", playerType: "batter" },
    },
    result: {
      data: {
        projectionBoard: [
          {
            playerId: 1,
            playerName: "Mike Trout",
            system: "steamer",
            version: "2026",
            sourceType: "first_party",
            playerType: "batter",
            stats: { pa: 600, hr: 35, rbi: 90, avg: 0.283 },
          },
          {
            playerId: 2,
            playerName: "Shohei Ohtani",
            system: "steamer",
            version: "2026",
            sourceType: "first_party",
            playerType: "batter",
            stats: { pa: 550, hr: 40, rbi: 95, avg: 0.275 },
          },
        ],
      },
    },
  };
}

function renderView(mocks: MockedResponse[] = [webConfigMock(), leagueMock(), boardMock()]) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <PlayerDrawerProvider>
        <ProjectionsView season={2026} />
      </PlayerDrawerProvider>
    </MockedProvider>,
  );
}

describe("ProjectionsView", () => {
  afterEach(cleanup);

  it("renders heading and filter controls", () => {
    renderView();
    expect(screen.getByText("Projections")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Filter by name...")).toBeInTheDocument();
  });

  it("loads and displays projections with league stat columns", async () => {
    renderView();

    expect(await screen.findByText("Mike Trout")).toBeInTheDocument();
    expect(screen.getByText("Shohei Ohtani")).toBeInTheDocument();

    // League stat column headers
    expect(screen.getByText("HR")).toBeInTheDocument();
    expect(screen.getByText("RBI")).toBeInTheDocument();
    expect(screen.getByText("AVG")).toBeInTheDocument();
  });

  it("filters players by name", async () => {
    renderView();

    expect(await screen.findByText("Mike Trout")).toBeInTheDocument();
    const input = screen.getByPlaceholderText("Filter by name...");
    await act(() => userEvent.type(input, "trout"));

    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    expect(screen.queryByText("Shohei Ohtani")).not.toBeInTheDocument();
  });
});
