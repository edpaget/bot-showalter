import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import { PLAN_KEEPER_DRAFT_QUERY } from "../graphql/queries";
import { KeeperPlannerView } from "./KeeperPlannerView";

afterEach(cleanup);

function planMock(): MockedResponse {
  return {
    request: {
      query: PLAN_KEEPER_DRAFT_QUERY,
      variables: {
        season: 2026,
        maxKeepers: 5,
        boardPreviewSize: 20,
      },
    },
    result: {
      data: {
        planKeeperDraft: {
          scenarios: [
            {
              keeperIds: [1, 2],
              keepers: [
                {
                  playerId: 1,
                  playerName: "Mike Trout",
                  position: "OF",
                  cost: 10,
                  surplus: 25.0,
                  projectedValue: 35.0,
                  recommendation: "keep",
                },
                {
                  playerId: 2,
                  playerName: "Shohei Ohtani",
                  position: "OF",
                  cost: 15,
                  surplus: 15.0,
                  projectedValue: 30.0,
                  recommendation: "keep",
                },
              ],
              totalSurplus: 40.0,
              boardPreview: [
                {
                  playerId: 3,
                  playerName: "Gerrit Cole",
                  playerType: "pitcher",
                  position: "SP",
                  originalValue: 25.0,
                  adjustedValue: 28.0,
                  valueChange: 3.0,
                },
              ],
              scarcity: [
                {
                  position: "OF",
                  tier1Value: 30.0,
                  replacementValue: 5.0,
                  totalSurplus: 100.0,
                  dropoffSlope: -2.0,
                },
              ],
              categoryNeeds: [
                {
                  category: "Strikeouts",
                  currentRank: 8,
                  targetRank: 6,
                },
              ],
              strongestCategories: ["HR"],
              weakestCategories: ["SB"],
            },
            {
              keeperIds: [1],
              keepers: [
                {
                  playerId: 1,
                  playerName: "Mike Trout",
                  position: "OF",
                  cost: 10,
                  surplus: 25.0,
                  projectedValue: 35.0,
                  recommendation: "keep",
                },
              ],
              totalSurplus: 25.0,
              boardPreview: [],
              scarcity: [],
              categoryNeeds: [],
              strongestCategories: [],
              weakestCategories: [],
            },
          ],
        },
      },
    },
  };
}

function renderView(mocks: MockedResponse[] = [planMock()]) {
  return render(
    <MemoryRouter>
      <MockedProvider mocks={mocks} addTypename={false}>
        <PlayerDrawerProvider>
          <KeeperPlannerView />
        </PlayerDrawerProvider>
      </MockedProvider>
    </MemoryRouter>,
  );
}

describe("KeeperPlannerView", () => {
  it("renders scenarios after loading", async () => {
    const user = userEvent.setup();
    renderView();

    // Click load button
    await user.click(screen.getByRole("button", { name: "Load Scenarios" }));

    // Wait for scenario cards
    expect(await screen.findByText("Optimal")).toBeInTheDocument();
    expect(screen.getByText("Alternative 1")).toBeInTheDocument();

    // Keeper names in the detail panel (may appear in card summary too)
    expect(screen.getAllByText("Mike Trout").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Shohei Ohtani").length).toBeGreaterThanOrEqual(1);

    // Board preview
    expect(screen.getByText("Gerrit Cole")).toBeInTheDocument();

    // Category needs
    expect(screen.getByText("Strikeouts")).toBeInTheDocument();

    // Strengths and weaknesses badges
    expect(screen.getByText("HR")).toBeInTheDocument();
    expect(screen.getByText("SB")).toBeInTheDocument();
  });

  it("shows no-data message when empty result", async () => {
    const user = userEvent.setup();
    const emptyMock: MockedResponse = {
      request: {
        query: PLAN_KEEPER_DRAFT_QUERY,
        variables: { season: 2026, maxKeepers: 5, boardPreviewSize: 20 },
      },
      result: {
        data: { planKeeperDraft: { scenarios: [] } },
      },
    };
    renderView([emptyMock]);

    await user.click(screen.getByRole("button", { name: "Load Scenarios" }));
    expect(await screen.findByText(/No scenarios available/)).toBeInTheDocument();
  });

  it("shows error state", async () => {
    const user = userEvent.setup();
    const errorMock: MockedResponse = {
      request: {
        query: PLAN_KEEPER_DRAFT_QUERY,
        variables: {
          season: 2026,
          maxKeepers: 5,
          boardPreviewSize: 20,
        },
      },
      error: new Error("Network error"),
    };
    renderView([errorMock]);

    await user.click(screen.getByRole("button", { name: "Load Scenarios" }));
    expect(await screen.findByText(/Error:/)).toBeInTheDocument();
  });

  it("shows start draft button", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(screen.getByRole("button", { name: "Load Scenarios" }));
    expect(await screen.findByRole("button", { name: "Start Draft with This Set" })).toBeInTheDocument();
  });
});
