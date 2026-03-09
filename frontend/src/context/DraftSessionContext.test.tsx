import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, afterEach } from "vitest";
import { DraftSessionProvider, useDraftSession } from "./DraftSessionContext";
import type { PickResult, DraftState } from "../types/session";

function TestConsumer() {
  const ctx = useDraftSession();
  return (
    <div>
      <span data-testid="session-id">{ctx.sessionId ?? "none"}</span>
      <span data-testid="drafted-count">{ctx.draftedPlayerIds.size}</span>
      <span data-testid="recs-count">{ctx.recommendations.length}</span>
      <span data-testid="roster-count">{ctx.roster.length}</span>
      <span data-testid="needs-count">{ctx.needs.length}</span>
      <span data-testid="balance-count">{ctx.balance.length}</span>
      <button onClick={() => ctx.setSessionId(42)}>set-session</button>
      <button
        onClick={() => {
          const state: DraftState = {
            sessionId: 42,
            currentPick: 2,
            picks: [
              { pickNumber: 1, team: 1, playerId: 100, playerName: "Test Player", position: "OF", price: null },
            ],
            format: "snake",
            teams: 12,
            userTeam: 1,
            budgetRemaining: null,
          };
          ctx.setState(state);
        }}
      >
        set-state
      </button>
      <button
        onClick={() => {
          const result: PickResult = {
            pick: { pickNumber: 2, team: 1, playerId: 200, playerName: "New Player", position: "SP", price: null },
            state: {
              sessionId: 42,
              currentPick: 3,
              picks: [
                { pickNumber: 1, team: 1, playerId: 100, playerName: "Test Player", position: "OF", price: null },
                { pickNumber: 2, team: 1, playerId: 200, playerName: "New Player", position: "SP", price: null },
              ],
              format: "snake",
              teams: 12,
              userTeam: 1,
              budgetRemaining: null,
            },
            recommendations: [
              { playerId: 300, playerName: "Rec Player", position: "FIRST_BASE", value: 20, score: 0.8, reason: "Need 1B" },
            ],
            roster: [
              { pickNumber: 1, team: 1, playerId: 100, playerName: "Test Player", position: "OF", price: null },
              { pickNumber: 2, team: 1, playerId: 200, playerName: "New Player", position: "SP", price: null },
            ],
            needs: [{ position: "C", remaining: 1 }],
          };
          ctx.applyPickResult(result);
        }}
      >
        apply-pick
      </button>
      <button onClick={() => ctx.clearSession()}>clear</button>
    </div>
  );
}

describe("DraftSessionContext", () => {
  afterEach(cleanup);

  it("provides default empty state", () => {
    render(
      <DraftSessionProvider>
        <TestConsumer />
      </DraftSessionProvider>,
    );
    expect(screen.getByTestId("session-id")).toHaveTextContent("none");
    expect(screen.getByTestId("drafted-count")).toHaveTextContent("0");
  });

  it("updates session id", async () => {
    render(
      <DraftSessionProvider>
        <TestConsumer />
      </DraftSessionProvider>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByText("set-session"));
    expect(screen.getByTestId("session-id")).toHaveTextContent("42");
  });

  it("derives draftedPlayerIds from state picks", async () => {
    render(
      <DraftSessionProvider>
        <TestConsumer />
      </DraftSessionProvider>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByText("set-state"));
    expect(screen.getByTestId("drafted-count")).toHaveTextContent("1");
  });

  it("applyPickResult updates state, recommendations, roster, and needs", async () => {
    render(
      <DraftSessionProvider>
        <TestConsumer />
      </DraftSessionProvider>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByText("apply-pick"));
    expect(screen.getByTestId("drafted-count")).toHaveTextContent("2");
    expect(screen.getByTestId("recs-count")).toHaveTextContent("1");
    expect(screen.getByTestId("roster-count")).toHaveTextContent("2");
    expect(screen.getByTestId("needs-count")).toHaveTextContent("1");
  });

  it("clearSession resets all state", async () => {
    render(
      <DraftSessionProvider>
        <TestConsumer />
      </DraftSessionProvider>,
    );
    const user = userEvent.setup();
    await user.click(screen.getByText("set-session"));
    await user.click(screen.getByText("apply-pick"));
    await user.click(screen.getByText("clear"));
    expect(screen.getByTestId("session-id")).toHaveTextContent("none");
    expect(screen.getByTestId("drafted-count")).toHaveTextContent("0");
    expect(screen.getByTestId("recs-count")).toHaveTextContent("0");
  });
});
