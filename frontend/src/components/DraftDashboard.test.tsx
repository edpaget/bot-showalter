import { render, screen, act, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { describe, it, expect, afterEach } from "vitest";
import { DraftDashboard } from "./DraftDashboard";
import { DraftSessionProvider } from "../context/DraftSessionContext";
import { BOARD_QUERY, SESSIONS_QUERY, BALANCE_QUERY } from "../graphql/queries";
import { START_SESSION, PICK, UNDO, END_SESSION } from "../graphql/mutations";
import type { DraftBoardRow } from "../types/board";

function makeRow(overrides: Partial<DraftBoardRow> & { playerId: number }): DraftBoardRow {
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
    request: { query: BOARD_QUERY, variables: { season: 2026 } },
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
        },
      },
    },
  };
}

function pickMock(): MockedResponse {
  return {
    request: {
      query: PICK,
      variables: { sessionId: 1, playerId: 1, position: "OF" },
    },
    result: {
      data: {
        pick: {
          pick: { pickNumber: 1, team: 1, playerId: 1, playerName: "Mike Trout", position: "OF", price: null },
          state: {
            sessionId: 1,
            currentPick: 2,
            picks: [
              { pickNumber: 1, team: 1, playerId: 1, playerName: "Mike Trout", position: "OF", price: null },
            ],
            format: "snake",
            teams: 12,
            userTeam: 1,
            budgetRemaining: null,
          },
          recommendations: [
            { playerId: 2, playerName: "Gerrit Cole", position: "SP", value: 25, score: 0.9, reason: "Best value" },
          ],
          roster: [
            { pickNumber: 1, team: 1, playerId: 1, playerName: "Mike Trout", position: "OF", price: null },
          ],
          needs: [{ position: "C", remaining: 1 }, { position: "SP", remaining: 2 }],
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
          },
          recommendations: [
            { playerId: 1, playerName: "Mike Trout", position: "OF", value: 35, score: 0.95, reason: "Best value" },
          ],
          roster: [],
          needs: [{ position: "C", remaining: 1 }, { position: "OF", remaining: 3 }],
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

function balanceMock(sessionId: number = 1): MockedResponse {
  return {
    request: { query: BALANCE_QUERY, variables: { sessionId } },
    result: {
      data: {
        balance: [
          { category: "HR", projectedValue: 245, leagueRankEstimate: 3, strength: "strong" },
        ],
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
      <DraftSessionProvider>
        <DraftDashboard season={2026} />
      </DraftSessionProvider>
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
    renderDashboard([
      boardMock(),
      sessionsMock(),
      startSessionMock(),
      balanceMock(),
      boardMock(),
      pickMock(),
      balanceMock(),
    ]);
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
      balanceMock(),
      undoMock(),
      balanceMock(),
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
});
