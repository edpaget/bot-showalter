import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DraftSessionSummaryType, DraftStateType } from "../generated/graphql";
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

describe("SessionControls", () => {
  afterEach(cleanup);

  it("shows start form when no session active", () => {
    render(
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
    render(
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
    });
  });

  it("shows pick counter and controls when session is active", () => {
    render(
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
    render(
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
    render(
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
    render(
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
    render(
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
    render(
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
});
