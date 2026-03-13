import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { act } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TRADE_PICKS } from "../graphql/mutations";
import { EVALUATE_TRADE_QUERY } from "../graphql/queries";
import { TradeDialog } from "./TradeDialog";

async function tick() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

const defaultProps = {
  sessionId: 1,
  userTeam: 1,
  teams: 4,
  currentPick: 1,
  totalPicks: 8,
  trades: [],
  onTradeComplete: vi.fn(),
  onClose: vi.fn(),
};

function renderDialog(mocks: MockedResponse[] = [], overrides = {}) {
  const props = { ...defaultProps, ...overrides };
  return render(
    <MockedProvider mocks={mocks}>
      <TradeDialog {...props} />
    </MockedProvider>,
  );
}

describe("TradeDialog", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders two team selectors excluding each other", () => {
    renderDialog();
    const selects = screen.getAllByRole("combobox");
    expect(selects).toHaveLength(2);
    // Team A defaults to userTeam (1), so excludes team 1 from Team B options
    const teamBOptions = within(selects[1]!).getAllByRole("option");
    expect(teamBOptions.map((o) => o.textContent)).not.toContain("Team 1");
    // Team B defaults to 2, so excludes team 2 from Team A options
    const teamAOptions = within(selects[0]!).getAllByRole("option");
    expect(teamAOptions.map((o) => o.textContent)).not.toContain("Team 2");
  });

  it("shows correct remaining picks per team", () => {
    renderDialog();
    // Team A (team 1) has picks 1, 8 in a 4-team 8-pick draft
    expect(screen.getByText("Pick #1 (Rd 1)")).toBeInTheDocument();
    expect(screen.getByText("Pick #8 (Rd 2)")).toBeInTheDocument();
    // Team B (team 2, default) has picks 2, 7
    expect(screen.getByText("Pick #2 (Rd 1)")).toBeInTheDocument();
    expect(screen.getByText("Pick #7 (Rd 2)")).toBeInTheDocument();
  });

  it("evaluate button disabled when nothing selected", () => {
    renderDialog();
    expect(screen.getByText("Evaluate")).toBeDisabled();
    expect(screen.getByText("Execute Trade")).toBeDisabled();
  });

  it("evaluate calls query and displays results", async () => {
    const evalMock: MockedResponse = {
      request: {
        query: EVALUATE_TRADE_QUERY,
        variables: { sessionId: 1, gives: [1], receives: [2] },
      },
      result: {
        data: {
          evaluateTrade: {
            givesValue: 10.5,
            receivesValue: 8.2,
            netValue: -2.3,
            recommendation: "Unfavorable trade",
            givesDetail: [{ pickNumber: 1, value: 10.5 }],
            receivesDetail: [{ pickNumber: 2, value: 8.2 }],
          },
        },
      },
    };

    renderDialog([evalMock]);
    const user = userEvent.setup();

    // Select pick #1 to give (team A) and pick #2 to receive (team B)
    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[0]!); // Pick #1 (team A gives)
    await user.click(checkboxes[2]!); // Pick #2 (team B gives)

    await user.click(screen.getByText("Evaluate"));
    await tick();

    expect(screen.getByText("10.5")).toBeInTheDocument();
    expect(screen.getByText("8.2")).toBeInTheDocument();
    expect(screen.getByText("-2.3")).toBeInTheDocument();
    expect(screen.getByText("Unfavorable trade")).toBeInTheDocument();
  });

  it("execute calls mutation with teamA and invokes onTradeComplete", async () => {
    const onTradeComplete = vi.fn();
    const onClose = vi.fn();
    const newState = {
      sessionId: 1,
      currentPick: 1,
      picks: [],
      format: "snake",
      teams: 4,
      userTeam: 1,
      budgetRemaining: null,
      keeperCount: 0,
      trades: [{ teamA: 1, teamB: 2, teamAGives: [1], teamBGives: [2] }],
    };
    const tradeMock: MockedResponse = {
      request: {
        query: TRADE_PICKS,
        variables: { sessionId: 1, gives: [1], receives: [2], partnerTeam: 2, teamA: 1 },
      },
      result: { data: { tradePicks: newState } },
    };

    renderDialog([tradeMock], { onTradeComplete, onClose });
    const user = userEvent.setup();

    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[0]!); // Pick #1 (team A gives)
    await user.click(checkboxes[2]!); // Pick #2 (team B gives)

    await user.click(screen.getByText("Execute Trade"));
    await tick();

    expect(onTradeComplete).toHaveBeenCalledWith(newState);
    expect(onClose).toHaveBeenCalled();
  });

  it("changing team B clears its selections", async () => {
    renderDialog();
    const user = userEvent.setup();

    // Select a team B pick
    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[2]!); // Pick #2 (team B gives)
    expect(checkboxes[2]).toBeChecked();

    // Change team B to team 3
    const selects = screen.getAllByRole("combobox");
    await user.selectOptions(selects[1]!, "3");

    // New checkboxes should be unchecked
    const newCheckboxes = screen.getAllByRole("checkbox");
    for (const cb of newCheckboxes.slice(2)) {
      expect(cb).not.toBeChecked();
    }
  });

  it("supports trades between non-user teams", async () => {
    const onTradeComplete = vi.fn();
    const onClose = vi.fn();
    const newState = {
      sessionId: 1,
      currentPick: 1,
      picks: [],
      format: "snake",
      teams: 4,
      userTeam: 1,
      budgetRemaining: null,
      keeperCount: 0,
      trades: [{ teamA: 3, teamB: 4, teamAGives: [3], teamBGives: [4] }],
    };
    const tradeMock: MockedResponse = {
      request: {
        query: TRADE_PICKS,
        variables: { sessionId: 1, gives: [3], receives: [4], partnerTeam: 4, teamA: 3 },
      },
      result: { data: { tradePicks: newState } },
    };

    renderDialog([tradeMock], { onTradeComplete, onClose });
    const user = userEvent.setup();

    // Change Team A to team 3
    const selects = screen.getAllByRole("combobox");
    await user.selectOptions(selects[0]!, "3");
    // Change Team B to team 4
    await user.selectOptions(selects[1]!, "4");

    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[0]!); // Pick #3 (team 3 gives)
    await user.click(checkboxes[2]!); // Pick #4 (team 4 gives)

    await user.click(screen.getByText("Execute Trade"));
    await tick();

    expect(onTradeComplete).toHaveBeenCalledWith(newState);
    expect(onClose).toHaveBeenCalled();
  });

  it("cancel calls onClose", async () => {
    const onClose = vi.fn();
    renderDialog([], { onClose });
    const user = userEvent.setup();

    await user.click(screen.getByText("Cancel"));
    expect(onClose).toHaveBeenCalled();
  });

  it("renders team names in dropdowns when provided", () => {
    const teamNames = { 1: "Sluggers", 2: "Aces", 3: "Dingers", 4: "Bombers" };
    renderDialog([], { teamNames });
    const selects = screen.getAllByRole("combobox");
    // Team A dropdown should have team names
    const teamAOptions = within(selects[0]!).getAllByRole("option");
    expect(teamAOptions.map((o) => o.textContent)).toContain("Sluggers");
    expect(teamAOptions.map((o) => o.textContent)).not.toContain("Aces"); // excluded (it's team B default)
    // Team B dropdown should have team names
    const teamBOptions = within(selects[1]!).getAllByRole("option");
    expect(teamBOptions.map((o) => o.textContent)).toContain("Aces");
    expect(teamBOptions.map((o) => o.textContent)).not.toContain("Sluggers"); // excluded (it's team A default)
  });

  it("falls back to Team N without team names", () => {
    renderDialog();
    const selects = screen.getAllByRole("combobox");
    const teamAOptions = within(selects[0]!).getAllByRole("option");
    expect(teamAOptions.some((o) => o.textContent === "Team 1")).toBe(true);
  });
});
