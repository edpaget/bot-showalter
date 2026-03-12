import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import { PLAN_KEEPER_DRAFT_QUERY, WEB_CONFIG_QUERY, YAHOO_KEEPER_OVERVIEW_QUERY } from "../graphql/queries";
import { KeeperPlannerView } from "./KeeperPlannerView";

afterEach(cleanup);

function webConfigMock(yahooLeague: object | null = null): MockedResponse {
  return {
    request: { query: WEB_CONFIG_QUERY },
    result: {
      data: {
        webConfig: {
          projections: [],
          valuations: [],
          yahooLeague,
        },
      },
    },
  };
}

const YAHOO_LEAGUE = {
  leagueKey: "449.l.12345",
  leagueName: "Test League",
  season: 2026,
  numTeams: 12,
  isKeeper: true,
  maxKeepers: 3,
  userTeamName: "My Team",
};

function planMock(maxKeepers = 5): MockedResponse {
  return {
    request: {
      query: PLAN_KEEPER_DRAFT_QUERY,
      variables: {
        season: 2026,
        maxKeepers,
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

function overviewMock(): MockedResponse {
  return {
    request: {
      query: YAHOO_KEEPER_OVERVIEW_QUERY,
      variables: { leagueKey: "449.l.12345", season: 2026, maxKeepers: 3 },
    },
    result: {
      data: {
        yahooKeeperOverview: {
          teamProjections: [
            {
              teamKey: "449.l.12345.t.1",
              teamName: "My Team",
              isUser: true,
              totalValue: 40.0,
              categoryTotals: { HR: 5.0 },
              keepers: [
                {
                  playerId: 1,
                  playerName: "Mike Trout",
                  position: "OF",
                  value: 35.0,
                  categoryScores: { HR: 2.5 },
                },
              ],
            },
            {
              teamKey: "449.l.12345.t.2",
              teamName: "Rival Squad",
              isUser: false,
              totalValue: 30.0,
              categoryTotals: { HR: 3.0 },
              keepers: [
                {
                  playerId: 2,
                  playerName: "Shohei Ohtani",
                  position: "OF",
                  value: 30.0,
                  categoryScores: { HR: 2.0 },
                },
              ],
            },
          ],
          tradeTargets: [],
          categoryNames: ["HR", "RBI"],
        },
      },
    },
  };
}

function renderView(mocks: MockedResponse[] = [webConfigMock(), planMock()]) {
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
    renderView([webConfigMock(), emptyMock]);

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
    renderView([webConfigMock(), errorMock]);

    await user.click(screen.getByRole("button", { name: "Load Scenarios" }));
    expect(await screen.findByText(/Error:/)).toBeInTheDocument();
  });

  it("shows start draft button", async () => {
    const user = userEvent.setup();
    renderView();

    await user.click(screen.getByRole("button", { name: "Load Scenarios" }));
    expect(await screen.findByRole("button", { name: "Start Draft with This Set" })).toBeInTheDocument();
  });

  it("shows sync button when Yahoo league is configured", async () => {
    renderView([webConfigMock(YAHOO_LEAGUE), planMock(3)]);
    expect(await screen.findByRole("button", { name: "Sync Keeper Costs from Yahoo" })).toBeInTheDocument();
  });

  it("does not show sync button without Yahoo league", () => {
    renderView([webConfigMock(), planMock()]);
    expect(screen.queryByRole("button", { name: "Sync Keeper Costs from Yahoo" })).not.toBeInTheDocument();
  });

  it("auto-populates maxKeepers from Yahoo league config", async () => {
    renderView([webConfigMock(YAHOO_LEAGUE), planMock(3)]);
    // Wait for config to load and effect to run
    const input = await screen.findByRole("button", { name: "Sync Keeper Costs from Yahoo" });
    expect(input).toBeInTheDocument();
    // The max keepers input should have value 3 from YAHOO_LEAGUE.maxKeepers
    const maxKeepersInput = screen.getAllByRole("spinbutton")[1];
    expect(maxKeepersInput).toHaveValue(3);
  });

  it("shows other teams keepers section when Yahoo is configured", async () => {
    renderView([webConfigMock(YAHOO_LEAGUE), planMock(3), overviewMock()]);
    // Wait for config to load
    expect(await screen.findByText("Show Other Teams' Keepers")).toBeInTheDocument();
  });

  it("renders team projections when other teams section is expanded", async () => {
    const user = userEvent.setup();
    renderView([webConfigMock(YAHOO_LEAGUE), planMock(3), overviewMock()]);

    // Click to show other teams
    const toggle = await screen.findByText("Show Other Teams' Keepers");
    await user.click(toggle);

    // Should load and display team projections
    expect(await screen.findByText("Rival Squad")).toBeInTheDocument();
    expect(screen.getByText("(You)")).toBeInTheDocument();
  });

  it("does not show other teams section without Yahoo league", () => {
    renderView([webConfigMock(), planMock()]);
    expect(screen.queryByText("Show Other Teams' Keepers")).not.toBeInTheDocument();
  });
});
