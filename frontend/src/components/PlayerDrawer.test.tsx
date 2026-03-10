import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerDrawerProvider, usePlayerDrawer } from "../context/PlayerDrawerContext";
import { LEAGUE_QUERY, PLAYER_BIO_QUERY, PROJECTIONS_QUERY, VALUATIONS_QUERY } from "../graphql/queries";
import { PlayerDrawer } from "./PlayerDrawer";

function bioMock(): MockedResponse {
  return {
    request: {
      query: PLAYER_BIO_QUERY,
      variables: { playerId: 1, season: 2026 },
    },
    result: {
      data: {
        playerBio: {
          playerId: 1,
          name: "Mike Trout",
          team: "LAA",
          age: 34,
          primaryPosition: "CF",
          bats: "R",
          throws: "R",
          experience: 14,
        },
      },
    },
  };
}

function projMock(): MockedResponse {
  return {
    request: {
      query: PROJECTIONS_QUERY,
      variables: { season: 2026, playerName: "Mike Trout" },
    },
    result: {
      data: {
        projections: [
          {
            playerName: "Mike Trout",
            system: "steamer",
            version: "2026",
            sourceType: "first_party",
            playerType: "batter",
            stats: { pa: 600, hr: 35 },
          },
        ],
      },
    },
  };
}

function valMock(): MockedResponse {
  return {
    request: {
      query: VALUATIONS_QUERY,
      variables: { season: 2026, top: 500 },
    },
    result: {
      data: {
        valuations: [
          {
            playerName: "Mike Trout",
            system: "zar",
            version: "1.0",
            projectionSystem: "steamer",
            projectionVersion: "2026",
            playerType: "batter",
            position: "OF",
            value: 35.0,
            rank: 1,
            categoryScores: { HR: 2.5 },
          },
        ],
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
          name: "H2H",
          format: "h2h_categories",
          teams: 12,
          budget: 260,
          rosterBatters: 9,
          rosterPitchers: 8,
          rosterUtil: 1,
          battingCategories: [
            { key: "hr", name: "HR", statType: "counting", direction: "higher" },
            { key: "r", name: "R", statType: "counting", direction: "higher" },
          ],
          pitchingCategories: [
            { key: "era", name: "ERA", statType: "rate", direction: "lower" },
            { key: "so", name: "K", statType: "counting", direction: "higher" },
          ],
        },
      },
    },
  };
}

function OpenButton() {
  const { openPlayer } = usePlayerDrawer();
  return (
    <button type="button" onClick={() => openPlayer(1, "Mike Trout")}>
      Open Drawer
    </button>
  );
}

function renderDrawer(mocks: MockedResponse[] = [bioMock(), projMock(), valMock(), leagueMock()]) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <PlayerDrawerProvider season={2026}>
        <OpenButton />
        <PlayerDrawer />
      </PlayerDrawerProvider>
    </MockedProvider>,
  );
}

describe("PlayerDrawer", () => {
  afterEach(cleanup);

  it("does not render when closed", () => {
    renderDrawer();
    expect(screen.queryByText("Biography")).not.toBeInTheDocument();
  });

  it("renders bio, projections, and valuations when open", async () => {
    renderDrawer();
    await act(() => userEvent.click(screen.getByText("Open Drawer")));

    // Should show player name
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    // Should show section headers
    expect(screen.getByText("Biography")).toBeInTheDocument();
    expect(screen.getByText("Projections")).toBeInTheDocument();
    expect(screen.getByText("Valuations")).toBeInTheDocument();
  });

  it("closes when backdrop is clicked", async () => {
    renderDrawer();
    await act(() => userEvent.click(screen.getByText("Open Drawer")));
    expect(screen.getByText("Biography")).toBeInTheDocument();

    await act(() => userEvent.click(screen.getByTestId("drawer-backdrop")));
    expect(screen.queryByText("Biography")).not.toBeInTheDocument();
  });

  it("filters projections by player type and shows league category columns", async () => {
    const mixedProjMock: MockedResponse = {
      request: {
        query: PROJECTIONS_QUERY,
        variables: { season: 2026, playerName: "Mike Trout" },
      },
      result: {
        data: {
          projections: [
            {
              playerName: "Mike Trout",
              system: "steamer",
              version: "2026",
              sourceType: "first_party",
              playerType: "batter",
              stats: { pa: 600, hr: 35, r: 90 },
            },
            {
              playerName: "Mike Trout",
              system: "steamer",
              version: "2026",
              sourceType: "first_party",
              playerType: "pitcher",
              stats: { ip: 10, era: 4.5, so: 5 },
            },
          ],
        },
      },
    };

    renderDrawer([bioMock(), mixedProjMock, valMock(), leagueMock()]);
    await act(() => userEvent.click(screen.getByText("Open Drawer")));

    // Wait for bio to load (which determines player type)
    expect(await screen.findByText("CF")).toBeInTheDocument();

    // Should show batter projection but not pitcher projection
    // Only 1 steamer row (batter), the pitcher one is filtered out
    const rows = screen.getAllByRole("row");
    const projRows = rows.filter((r) => r.textContent?.includes("steamer"));
    expect(projRows.length).toBe(1);
    // The batter row should show HR stat value, not pitcher stats
    expect(projRows[0]!.textContent).toContain("35");
    expect(projRows[0]!.textContent).toContain("90");

    // League category headers should be present for batting (HR, R)
    expect(screen.getByRole("columnheader", { name: "HR" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "R" })).toBeInTheDocument();
  });
});
