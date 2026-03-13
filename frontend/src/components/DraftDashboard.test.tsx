import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { act, cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { DraftSessionProvider } from "../context/DraftSessionContext";
import { PlayerDrawerProvider } from "../context/PlayerDrawerContext";
import type { DraftBoardRowType } from "../generated/graphql";
import { END_SESSION, PICK, START_SESSION, UNDO } from "../graphql/mutations";
import { BALANCE_QUERY, BOARD_QUERY, LEAGUE_QUERY, SESSIONS_QUERY } from "../graphql/queries";
import { DraftDashboard } from "./DraftDashboard";

function makeRow(overrides: Partial<DraftBoardRowType> & { playerId: number }): DraftBoardRowType {
  return {
    playerName: `Player ${overrides.playerId}`,
    rank: overrides.playerId,
    playerType: "batter",
    position: "OF",
    value: 10,
    categoryZScores: {},
    age: null,
    batsThrows: null,
    tier: null,
    adpOverall: null,
    adpRank: null,
    adpDelta: null,
    breakoutRank: null,
    bustRank: null,
    ...overrides,
  };
}

const BOARD_ROWS = [
  makeRow({ playerId: 1, playerName: "Mike Trout", rank: 1, position: "OF", value: 35 }),
  makeRow({ playerId: 2, playerName: "Gerrit Cole", rank: 2, playerType: "pitcher", position: "SP", value: 25 }),
];

function boardMock(): MockedResponse {
  return {
    request: { query: BOARD_QUERY, variables: { season: 2026, system: null, version: null } },
    result: {
      data: {
        board: {
          rows: BOARD_ROWS,
          battingCategories: ["HR", "RBI"],
          pitchingCategories: ["W", "K"],
        },
      },
    },
  };
}

function sessionsMock(sessions: unknown[] = []): MockedResponse {
  return {
    request: { query: SESSIONS_QUERY, variables: { status: "active" } },
    result: { data: { sessions } },
  };
}

function startSessionMock(): MockedResponse {
  return {
    request: {
      query: START_SESSION,
      variables: { season: 2026, teams: 12, format: "snake", userTeam: 1 },
    },
    result: {
      data: {
        startSession: {
          sessionId: 1,
          currentPick: 1,
          picks: [],
          format: "snake",
          teams: 12,
          userTeam: 1,
          budgetRemaining: null,
          keeperCount: 0,
          trades: [],
        },
      },
    },
  };
}

function pickMock(): MockedResponse {
  return {
    request: {
      query: PICK,
      variables: { sessionId: 1, playerId: 1, position: "OF", playerType: "batter" },
    },
    result: {
      data: {
        pick: {
          pick: { pickNumber: 1, team: 1, playerId: 1, playerName: "Mike Trout", position: "OF", price: null },
          state: {
            sessionId: 1,
            currentPick: 2,
            picks: [{ pickNumber: 1, team: 1, playerId: 1, playerName: "Mike Trout", position: "OF", price: null }],
            format: "snake",
            teams: 12,
            userTeam: 1,
            budgetRemaining: null,
            keeperCount: 0,
            trades: [],
          },
          recommendations: [
            {
              playerId: 2,
              playerName: "Gerrit Cole",
              position: "SP",
              value: 25,
              score: 0.9,
              reason: "Best value",
              playerType: "pitcher",
            },
          ],
          roster: [{ pickNumber: 1, team: 1, playerId: 1, playerName: "Mike Trout", position: "OF", price: null }],
          needs: [
            { position: "C", remaining: 1 },
            { position: "SP", remaining: 2 },
          ],
          arbitrage: null,
          balance: [],
          categoryNeeds: [],
        },
      },
    },
  };
}

function undoMock(): MockedResponse {
  return {
    request: { query: UNDO, variables: { sessionId: 1 } },
    result: {
      data: {
        undo: {
          pick: { pickNumber: 1, team: 1, playerId: 1, playerName: "Mike Trout", position: "OF", price: null },
          state: {
            sessionId: 1,
            currentPick: 1,
            picks: [],
            format: "snake",
            teams: 12,
            userTeam: 1,
            budgetRemaining: null,
            keeperCount: 0,
            trades: [],
          },
          recommendations: [
            {
              playerId: 1,
              playerName: "Mike Trout",
              position: "OF",
              value: 35,
              score: 0.95,
              reason: "Best value",
              playerType: "batter",
            },
          ],
          roster: [],
          needs: [
            { position: "C", remaining: 1 },
            { position: "OF", remaining: 3 },
          ],
          arbitrage: null,
          balance: [],
          categoryNeeds: [],
        },
      },
    },
  };
}

function endSessionMock(): MockedResponse {
  return {
    request: { query: END_SESSION, variables: { sessionId: 1 } },
    result: { data: { endSession: true } },
  };
}

function leagueMock(): MockedResponse {
  return {
    request: { query: LEAGUE_QUERY },
    result: {
      data: {
        league: {
          name: "Test League",
          format: "snake",
          teams: 12,
          budget: 260,
          rosterBatters: 10,
          rosterPitchers: 9,
          rosterUtil: 1,
          battingCategories: [],
          pitchingCategories: [],
        },
      },
    },
  };
}

function balanceMock(sessionId: number = 1): MockedResponse {
  return {
    request: { query: BALANCE_QUERY, variables: { sessionId } },
    result: {
      data: {
        balance: [{ category: "HR", projectedValue: 245, leagueRankEstimate: 3, strength: "strong" }],
      },
    },
  };
}

async function tick() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

function renderDashboard(mocks: MockedResponse[]) {
  render(
    <MockedProvider mocks={mocks}>
      <PlayerDrawerProvider>
        <DraftSessionProvider>
          <DraftDashboard season={2026} />
        </DraftSessionProvider>
      </PlayerDrawerProvider>
    </MockedProvider>,
  );
}

describe("DraftDashboard", () => {
  afterEach(cleanup);

  it("renders board and start form when no session active", async () => {
    renderDashboard([boardMock(), sessionsMock()]);
    await tick();
    expect(screen.getByText("Start Draft")).toBeInTheDocument();
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
  });

  it("starts a session and shows sidebar panels", async () => {
    renderDashboard([
      boardMock(),
      sessionsMock(),
      startSessionMock(),
      balanceMock(),
      // Board is re-fetched after session start due to sessionActive changing
      boardMock(),
    ]);
    await tick();

    const user = userEvent.setup();
    await user.click(screen.getByText("Start Draft"));
    await tick();

    expect(screen.getByText("Pick #1")).toBeInTheDocument();
    expect(screen.getByText("Recommendations")).toBeInTheDocument();
    expect(screen.getByText("Roster")).toBeInTheDocument();
    expect(screen.getByText("Needs")).toBeInTheDocument();
  });

  it("picks a player and updates panels", async () => {
    renderDashboard([boardMock(), sessionsMock(), startSessionMock(), balanceMock(), boardMock(), pickMock()]);
    await tick();

    const user = userEvent.setup();
    // Start session
    await user.click(screen.getByText("Start Draft"));
    await tick();

    // Pick Trout via board Draft button
    const draftButtons = screen.getAllByRole("button", { name: "Draft" });
    await user.click(draftButtons[0]!);
    await tick();

    // Verify panels updated
    expect(screen.getByText("Pick #2")).toBeInTheDocument();
    // Roster panel should have 1 player
    expect(screen.getByText("Players: 1")).toBeInTheDocument();
  });

  it("undoes a pick", async () => {
    renderDashboard([
      boardMock(),
      sessionsMock(),
      startSessionMock(),
      balanceMock(),
      boardMock(),
      pickMock(),
      undoMock(),
    ]);
    await tick();

    const user = userEvent.setup();
    await user.click(screen.getByText("Start Draft"));
    await tick();

    // Pick
    const draftButtons = screen.getAllByRole("button", { name: "Draft" });
    await user.click(draftButtons[0]!);
    await tick();

    // Undo
    await user.click(screen.getByText("Undo"));
    await tick();

    expect(screen.getByText("Pick #1")).toBeInTheDocument();
  });

  it("ends a session and returns to start form", async () => {
    renderDashboard([
      boardMock(),
      sessionsMock(),
      startSessionMock(),
      balanceMock(),
      boardMock(),
      endSessionMock(),
      sessionsMock(),
      boardMock(),
    ]);
    await tick();

    const user = userEvent.setup();
    await user.click(screen.getByText("Start Draft"));
    await tick();

    await user.click(screen.getByText("End Session"));
    await tick();

    expect(screen.getByText("Start Draft")).toBeInTheDocument();
  });

  it("shows Trade Picks button for snake format sessions", async () => {
    renderDashboard([boardMock(), sessionsMock(), startSessionMock(), balanceMock(), leagueMock(), boardMock()]);
    await tick();

    const user = userEvent.setup();
    await user.click(screen.getByText("Start Draft"));
    await tick();

    expect(screen.getByText("Trade Picks")).toBeInTheDocument();
  });

  it("hides Trade Picks button for auction format sessions", async () => {
    const auctionStartMock: MockedResponse = {
      request: {
        query: START_SESSION,
        variables: { season: 2026, teams: 12, format: "auction", userTeam: 1, budget: 260 },
      },
      result: {
        data: {
          startSession: {
            sessionId: 2,
            currentPick: 1,
            picks: [],
            format: "auction",
            teams: 12,
            userTeam: 1,
            budgetRemaining: 260,
            keeperCount: 0,
            trades: [],
          },
        },
      },
    };

    renderDashboard([boardMock(), sessionsMock(), auctionStartMock, balanceMock(2), boardMock()]);
    await tick();

    const user = userEvent.setup();
    // Change format to auction
    const formatSelect = screen.getByDisplayValue("Snake");
    await user.selectOptions(formatSelect, "auction");
    await tick();

    await user.click(screen.getByText("Start Draft"));
    await tick();

    expect(screen.queryByText("Trade Picks")).not.toBeInTheDocument();
  });
});
