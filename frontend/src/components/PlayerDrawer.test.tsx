import { render, screen, act, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { describe, it, expect, afterEach } from "vitest";
import { PlayerDrawer } from "./PlayerDrawer";
import { PlayerDrawerProvider, usePlayerDrawer } from "../context/PlayerDrawerContext";
import { PLAYER_BIO_QUERY, PROJECTIONS_QUERY, VALUATIONS_QUERY } from "../graphql/queries";

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

function OpenButton() {
  const { openPlayer } = usePlayerDrawer();
  return <button onClick={() => openPlayer(1, "Mike Trout")}>Open Drawer</button>;
}

function renderDrawer(mocks: MockedResponse[] = [bioMock(), projMock(), valMock()]) {
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

  it("filters projections by player type and shows version column", async () => {
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
              stats: { pa: 600, hr: 35 },
            },
            {
              playerName: "Mike Trout",
              system: "steamer",
              version: "2026",
              sourceType: "first_party",
              playerType: "pitcher",
              stats: { ip: 10, k: 5 },
            },
          ],
        },
      },
    };

    renderDrawer([bioMock(), mixedProjMock, valMock()]);
    await act(() => userEvent.click(screen.getByText("Open Drawer")));

    // Wait for bio to load (which determines player type)
    expect(await screen.findByText("CF")).toBeInTheDocument();

    // Should show batter projection but not pitcher projection
    // The batter row should be visible, pitcher row filtered out
    const rows = screen.getAllByRole("row");
    const projRows = rows.filter((r) => r.textContent?.includes("steamer"));
    // Only the batter projection should appear (1 row), not the pitcher one
    expect(projRows.length).toBe(1);
    expect(projRows[0].textContent).toContain("batter");
    expect(projRows[0].textContent).not.toContain("pitcher");

    // Version columns should be present (projections + valuations tables)
    expect(screen.getAllByText("Version").length).toBeGreaterThanOrEqual(1);
  });
});
