import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GraphQLError } from "graphql";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DraftSessionSummaryType, DraftStateType } from "../generated/graphql";
import { YAHOO_DRAFT_SETUP_QUERY } from "../graphql/queries";
import { SessionControls } from "./SessionControls";

const ACTIVE_STATE: DraftStateType = {
  sessionId: 1,
  currentPick: 5,
  picks: [{ pickNumber: 1, team: 1, playerId: 100, playerName: "Mike Trout", position: "OF", price: null }],
  format: "snake",
  teams: 12,
  userTeam: 1,
  budgetRemaining: null,
  keeperCount: 0,
  teamNames: null,
  trades: [],
};

const SESSIONS: DraftSessionSummaryType[] = [
  {
    id: 1,
    league: "test",
    season: 2026,
    teams: 12,
    format: "snake",
    userTeam: 1,
    status: "active",
    pickCount: 24,
    createdAt: "2026-03-09",
    updatedAt: "2026-03-09",
    system: "gbm",
    version: "v1",
  },
];

const YAHOO_LEAGUE = { leagueKey: "mlb.l.12345", leagueName: "Test League", season: 2026 };

function yahooDraftSetupMock(overrides?: Partial<MockedResponse>): MockedResponse {
  return {
    request: { query: YAHOO_DRAFT_SETUP_QUERY, variables: { leagueKey: "mlb.l.12345", season: 2026 } },
    result: {
      data: {
        yahooDraftSetup: {
          numTeams: 10,
          draftFormat: "snake",
          userTeamId: 3,
          teamNames: {},
          draftOrder: [],
          isKeeper: true,
          maxKeepers: 5,
          keeperPlayerIds: [101, 202],
        },
      },
    },
    ...overrides,
  };
}

function yahooDraftSetupErrorMock(): MockedResponse {
  return {
    request: { query: YAHOO_DRAFT_SETUP_QUERY, variables: { leagueKey: "mlb.l.12345", season: 2026 } },
    result: { errors: [new GraphQLError("League not synced")] },
  };
}

function renderWithApollo(ui: React.ReactElement, mocks: MockedResponse[] = []) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      {ui}
    </MockedProvider>,
  );
}

describe("SessionControls", () => {
  afterEach(cleanup);

  it("shows start form when no session active", () => {
    renderWithApollo(
      <SessionControls
        sessionActive={false}
        state={null}
        sessions={[]}
        onStart={vi.fn()}
        onResume={vi.fn()}
        onUndo={vi.fn()}
        onEnd={vi.fn()}
      />,
    );
    expect(screen.getByText("Start Draft")).toBeInTheDocument();
  });

  it("calls onStart with config", async () => {
    const onStart = vi.fn();
    renderWithApollo(
      <SessionControls
        sessionActive={false}
        state={null}
        sessions={[]}
        onStart={onStart}
        onResume={vi.fn()}
        onUndo={vi.fn()}
        onEnd={vi.fn()}
      />,
    );
    const user = userEvent.setup();
    await user.click(screen.getByText("Start Draft"));
    expect(onStart).toHaveBeenCalledWith({
      season: 2026,
      teams: 12,
      format: "snake",
      userTeam: 1,
      budget: undefined,
      keeperPlayerIds: undefined,
    });
  });

  it("shows pick counter and controls when session is active", () => {
    renderWithApollo(
      <SessionControls
        sessionActive
        state={ACTIVE_STATE}
        sessions={[]}
        onStart={vi.fn()}
        onResume={vi.fn()}
        onUndo={vi.fn()}
        onEnd={vi.fn()}
      />,
    );
    expect(screen.getByText("Pick #5")).toBeInTheDocument();
    expect(screen.getByText("Undo")).toBeInTheDocument();
    expect(screen.getByText("End Session")).toBeInTheDocument();
  });

  it("calls onUndo when Undo is clicked", async () => {
    const onUndo = vi.fn();
    renderWithApollo(
      <SessionControls
        sessionActive
        state={ACTIVE_STATE}
        sessions={[]}
        onStart={vi.fn()}
        onResume={vi.fn()}
        onUndo={onUndo}
        onEnd={vi.fn()}
      />,
    );
    const user = userEvent.setup();
    await user.click(screen.getByText("Undo"));
    expect(onUndo).toHaveBeenCalled();
  });

  it("calls onEnd when End Session is clicked", async () => {
    const onEnd = vi.fn();
    renderWithApollo(
      <SessionControls
        sessionActive
        state={ACTIVE_STATE}
        sessions={[]}
        onStart={vi.fn()}
        onResume={vi.fn()}
        onUndo={vi.fn()}
        onEnd={onEnd}
      />,
    );
    const user = userEvent.setup();
    await user.click(screen.getByText("End Session"));
    expect(onEnd).toHaveBeenCalled();
  });

  it("shows resume buttons for active sessions", () => {
    renderWithApollo(
      <SessionControls
        sessionActive={false}
        state={null}
        sessions={SESSIONS}
        onStart={vi.fn()}
        onResume={vi.fn()}
        onUndo={vi.fn()}
        onEnd={vi.fn()}
      />,
    );
    expect(screen.getByText("#1 — 2026 snake (24 picks)")).toBeInTheDocument();
  });

  it("calls onResume when resume button is clicked", async () => {
    const onResume = vi.fn();
    renderWithApollo(
      <SessionControls
        sessionActive={false}
        state={null}
        sessions={SESSIONS}
        onStart={vi.fn()}
        onResume={onResume}
        onUndo={vi.fn()}
        onEnd={vi.fn()}
      />,
    );
    const user = userEvent.setup();
    await user.click(screen.getByText("#1 — 2026 snake (24 picks)"));
    expect(onResume).toHaveBeenCalledWith(1);
  });

  it("disables undo when no picks", () => {
    const emptyState = { ...ACTIVE_STATE, picks: [] };
    renderWithApollo(
      <SessionControls
        sessionActive
        state={emptyState}
        sessions={[]}
        onStart={vi.fn()}
        onResume={vi.fn()}
        onUndo={vi.fn()}
        onEnd={vi.fn()}
      />,
    );
    expect(screen.getByText("Undo")).toBeDisabled();
  });

  describe("Yahoo prefill", () => {
    it("prefills form fields from Yahoo draft setup", async () => {
      renderWithApollo(
        <SessionControls
          sessionActive={false}
          state={null}
          sessions={[]}
          onStart={vi.fn()}
          onResume={vi.fn()}
          onUndo={vi.fn()}
          onEnd={vi.fn()}
          yahooLeague={YAHOO_LEAGUE}
        />,
        [yahooDraftSetupMock()],
      );

      await waitFor(() => {
        expect(screen.getByText(/Prefilled from Yahoo: Test League/)).toBeInTheDocument();
      });

      // Verify form values by finding inputs near their labels
      const inputs = screen.getAllByRole("spinbutton") as HTMLInputElement[];
      const teamsInput = inputs.find((i) => i.closest("div")?.textContent?.includes("Teams"));
      expect(teamsInput?.value).toBe("10");
      const userTeamInput = inputs.find((i) => i.closest("div")?.textContent?.includes("Your Team"));
      expect(userTeamInput?.value).toBe("3");
    });

    it("passes keeperPlayerIds through onStart", async () => {
      const onStart = vi.fn();
      renderWithApollo(
        <SessionControls
          sessionActive={false}
          state={null}
          sessions={[]}
          onStart={onStart}
          onResume={vi.fn()}
          onUndo={vi.fn()}
          onEnd={vi.fn()}
          yahooLeague={YAHOO_LEAGUE}
        />,
        [yahooDraftSetupMock()],
      );

      await waitFor(() => {
        expect(screen.getByText(/Prefilled from Yahoo/)).toBeInTheDocument();
      });

      const user = userEvent.setup();
      await user.click(screen.getByText("Start Draft"));

      expect(onStart).toHaveBeenCalledWith(
        expect.objectContaining({
          teams: 10,
          userTeam: 3,
          keeperPlayerIds: [101, 202],
        }),
      );
    });

    it("shows fallback message when Yahoo query fails", async () => {
      renderWithApollo(
        <SessionControls
          sessionActive={false}
          state={null}
          sessions={[]}
          onStart={vi.fn()}
          onResume={vi.fn()}
          onUndo={vi.fn()}
          onEnd={vi.fn()}
          yahooLeague={YAHOO_LEAGUE}
        />,
        [yahooDraftSetupErrorMock()],
      );

      await waitFor(() => {
        expect(screen.getByText(/Could not load Yahoo settings/)).toBeInTheDocument();
      });

      // Form still works with defaults
      expect(screen.getByText("Start Draft")).toBeInTheDocument();
    });

    it("does not prefill when yahooLeague is null", () => {
      renderWithApollo(
        <SessionControls
          sessionActive={false}
          state={null}
          sessions={[]}
          onStart={vi.fn()}
          onResume={vi.fn()}
          onUndo={vi.fn()}
          onEnd={vi.fn()}
          yahooLeague={null}
        />,
      );

      expect(screen.queryByText(/Prefilled from Yahoo/)).not.toBeInTheDocument();
      expect(screen.getByText("Start Draft")).toBeInTheDocument();
    });
  });
});
