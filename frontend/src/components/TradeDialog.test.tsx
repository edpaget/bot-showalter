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
    <MockedProvider mocks={mocks} addTypename={false}>
      <TradeDialog {...props} />
    </MockedProvider>,
  );
}

describe("TradeDialog", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders partner team options excluding user team", () => {
    renderDialog();
    const select = screen.getByRole("combobox");
    const options = within(select).getAllByRole("option");
    expect(options).toHaveLength(3);
    expect(options.map((o) => o.textContent)).toEqual(["Team 2", "Team 3", "Team 4"]);
  });

  it("shows correct remaining picks per team", () => {
    renderDialog();
    // User team 1 has picks 1, 8 in a 4-team 8-pick draft
    expect(screen.getByText("Pick #1 (Rd 1)")).toBeInTheDocument();
    expect(screen.getByText("Pick #8 (Rd 2)")).toBeInTheDocument();
    // Partner team 2 (default) has picks 2, 7
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

    // Select pick #1 to give
    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[0]!); // Pick #1 (give)
    await user.click(checkboxes[2]!); // Pick #2 (receive)

    await user.click(screen.getByText("Evaluate"));
    await tick();

    expect(screen.getByText("10.5")).toBeInTheDocument();
    expect(screen.getByText("8.2")).toBeInTheDocument();
    expect(screen.getByText("-2.3")).toBeInTheDocument();
    expect(screen.getByText("Unfavorable trade")).toBeInTheDocument();
  });

  it("execute calls mutation and invokes onTradeComplete", async () => {
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
        variables: { sessionId: 1, gives: [1], receives: [2], partnerTeam: 2 },
      },
      result: { data: { tradePicks: newState } },
    };

    renderDialog([tradeMock], { onTradeComplete, onClose });
    const user = userEvent.setup();

    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[0]!); // Pick #1 (give)
    await user.click(checkboxes[2]!); // Pick #2 (receive)

    await user.click(screen.getByText("Execute Trade"));
    await tick();

    expect(onTradeComplete).toHaveBeenCalledWith(newState);
    expect(onClose).toHaveBeenCalled();
  });

  it("changing partner team clears receive selections", async () => {
    renderDialog();
    const user = userEvent.setup();

    // Select a receive pick
    const checkboxes = screen.getAllByRole("checkbox");
    await user.click(checkboxes[2]!); // Pick #2 (receive for team 2)
    expect(checkboxes[2]).toBeChecked();

    // Change partner to team 3
    await user.selectOptions(screen.getByRole("combobox"), "3");

    // New receive checkboxes should be unchecked
    const newCheckboxes = screen.getAllByRole("checkbox");
    for (const cb of newCheckboxes.slice(2)) {
      expect(cb).not.toBeChecked();
    }
  });

  it("cancel calls onClose", async () => {
    const onClose = vi.fn();
    renderDialog([], { onClose });
    const user = userEvent.setup();

    await user.click(screen.getByText("Cancel"));
    expect(onClose).toHaveBeenCalled();
  });
});
