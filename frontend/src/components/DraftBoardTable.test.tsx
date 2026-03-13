import { MockedProvider, type MockedResponse } from "@apollo/client/testing";
import { act, cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DraftBoardRowType } from "../generated/graphql";
import { BOARD_QUERY } from "../graphql/queries";
import { DraftBoardTable } from "./DraftBoardTable";

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

const ROWS: DraftBoardRowType[] = [
  makeRow({
    playerId: 1,
    playerName: "Mike Trout",
    rank: 1,
    position: "OF",
    value: 35,
    tier: 1,
    adpDelta: 15,
    breakoutRank: 3,
  }),
  makeRow({
    playerId: 2,
    playerName: "Shohei Ohtani",
    rank: 2,
    position: "OF",
    value: 30,
    tier: 2,
    adpDelta: -12,
    bustRank: 5,
  }),
  makeRow({
    playerId: 3,
    playerName: "Gerrit Cole",
    rank: 3,
    playerType: "pitcher",
    position: "SP",
    value: 25,
    tier: 3,
  }),
  makeRow({ playerId: 4, playerName: "Aaron Judge", rank: 4, position: "OF", value: 20, tier: 1, breakoutRank: 1 }),
];

function boardMock(): MockedResponse {
  return {
    request: {
      query: BOARD_QUERY,
      variables: { season: 2026, system: null, version: null },
    },
    result: {
      data: {
        board: {
          rows: ROWS,
          battingCategories: ["HR", "RBI"],
          pitchingCategories: ["W", "K"],
        },
      },
    },
  };
}

async function renderAndWait(props?: Partial<React.ComponentProps<typeof DraftBoardTable>>) {
  render(
    <MockedProvider mocks={[boardMock()]}>
      <DraftBoardTable season={2026} {...props} />
    </MockedProvider>,
  );
  // Wait for Apollo mock to resolve (0ms delay by default, but need a tick)
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
}

function getColumnHeader(name: string) {
  return screen.getByRole("columnheader", { name: new RegExp(`^${name}`) });
}

function getDataRows() {
  return screen.getAllByRole("row").slice(1); // skip header row
}

describe("DraftBoardTable", () => {
  afterEach(cleanup);
  it("renders all players with correct columns", async () => {
    await renderAndWait();
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    expect(screen.getByText("Shohei Ohtani")).toBeInTheDocument();
    expect(screen.getByText("Gerrit Cole")).toBeInTheDocument();
    expect(screen.getByText("Aaron Judge")).toBeInTheDocument();
  });

  it("sorts by value when Value header is clicked", async () => {
    await renderAndWait();
    const user = userEvent.setup();
    await user.click(getColumnHeader("Value"));

    const rows = getDataRows();
    const values = rows.map((row) => within(row).getAllByRole("cell")[4]?.textContent);
    expect(values).toEqual(["$35.0", "$30.0", "$25.0", "$20.0"]);
  });

  it("sorts by rank desc when Rank header is clicked", async () => {
    await renderAndWait();
    const user = userEvent.setup();
    // Rank is the default sort (asc), clicking toggles to desc
    await user.click(getColumnHeader("Rank"));

    const rows = getDataRows();
    const ranks = rows.map((row) => within(row).getAllByRole("cell")[0]?.textContent);
    expect(ranks).toEqual(["4", "3", "2", "1"]);
  });

  it("filters by position when position button is clicked", async () => {
    await renderAndWait();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "SP" }));

    expect(screen.getByText("Gerrit Cole")).toBeInTheDocument();
    expect(screen.queryByText("Mike Trout")).not.toBeInTheDocument();
  });

  it("filters by player type when Pitchers tab is clicked", async () => {
    await renderAndWait();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Pitchers" }));

    expect(screen.getByText("Gerrit Cole")).toBeInTheDocument();
    expect(screen.queryByText("Mike Trout")).not.toBeInTheDocument();
  });

  it("filters by player name search", async () => {
    await renderAndWait();
    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText("Search player…"), "Trout");

    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    expect(screen.queryByText("Gerrit Cole")).not.toBeInTheDocument();
  });

  it("applies breakout tint when breakout rank is top-20", async () => {
    await renderAndWait();
    // Trout has breakoutRank=3 (top-20) so gets green tint instead of tier color
    const troutRow = screen.getByText("Mike Trout").closest("tr")!;
    expect(troutRow.style.backgroundColor).toBe("rgba(0, 128, 0, 0.06)");
  });

  it("applies tier color when no breakout/bust tint applies", async () => {
    await renderAndWait();
    // Cole: tier 3 (#fff3e0), no breakout/bust rank
    const coleRow = screen.getByText("Gerrit Cole").closest("tr")!;
    expect(coleRow.style.backgroundColor).toBe("rgb(255, 243, 224)");
  });

  it("shows breakout label B3", async () => {
    await renderAndWait();
    expect(screen.getByText("B3")).toBeInTheDocument();
  });

  it("shows bust label X5", async () => {
    await renderAndWait();
    expect(screen.getByText("X5")).toBeInTheDocument();
  });

  it("applies green tint for top-20 breakout player", async () => {
    await renderAndWait();
    const judgeRow = screen.getByText("Aaron Judge").closest("tr")!;
    expect(judgeRow.style.backgroundColor).toBe("rgba(0, 128, 0, 0.06)");
  });

  it("applies green class for ADP delta >= 10", async () => {
    await renderAndWait();
    const deltaCell = screen.getByText("15");
    expect(deltaCell).toHaveClass("text-green-700");
    expect(deltaCell).toHaveClass("font-bold");
  });

  it("applies red class for ADP delta <= -10", async () => {
    await renderAndWait();
    const deltaCell = screen.getByText("-12");
    expect(deltaCell).toHaveClass("text-red-700");
    expect(deltaCell).toHaveClass("font-bold");
  });

  it("shows Action column when session is active", async () => {
    await renderAndWait({ sessionActive: true, onDraft: vi.fn(), draftedPlayerKeys: new Set() });
    expect(screen.getByText("Action")).toBeInTheDocument();
  });

  it("hides Action column when session is not active", async () => {
    await renderAndWait();
    expect(screen.queryByText("Action")).not.toBeInTheDocument();
  });

  it("shows Draft buttons for undrafted players", async () => {
    await renderAndWait({ sessionActive: true, onDraft: vi.fn(), draftedPlayerKeys: new Set() });
    const draftButtons = screen.getAllByRole("button", { name: "Draft" });
    expect(draftButtons.length).toBe(4);
  });

  it("hides Draft button for drafted players", async () => {
    await renderAndWait({ sessionActive: true, onDraft: vi.fn(), draftedPlayerKeys: new Set(["1-batter"]) });
    // Trout (id=1) is drafted, so only 3 Draft buttons
    const draftButtons = screen.getAllByRole("button", { name: "Draft" });
    expect(draftButtons.length).toBe(3);
  });

  it("calls onDraft when Draft button clicked", async () => {
    const onDraft = vi.fn();
    await renderAndWait({ sessionActive: true, onDraft, draftedPlayerKeys: new Set() });
    const user = userEvent.setup();
    const draftButtons = screen.getAllByRole("button", { name: "Draft" });
    await user.click(draftButtons[0]!);
    expect(onDraft).toHaveBeenCalledWith(1, "OF");
  });

  it("grays out drafted players", async () => {
    await renderAndWait({ sessionActive: true, onDraft: vi.fn(), draftedPlayerKeys: new Set(["1-batter"]) });
    const troutRow = screen.getByText("Mike Trout").closest("tr")!;
    expect(troutRow.className).toContain("opacity-40");
  });

  it("filters by status (Available)", async () => {
    await renderAndWait({ sessionActive: true, onDraft: vi.fn(), draftedPlayerKeys: new Set(["1-batter"]) });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Available" }));
    expect(screen.queryByText("Mike Trout")).not.toBeInTheDocument();
    expect(screen.getByText("Shohei Ohtani")).toBeInTheDocument();
  });

  it("filters by status (Drafted)", async () => {
    await renderAndWait({ sessionActive: true, onDraft: vi.fn(), draftedPlayerKeys: new Set(["1-batter"]) });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Drafted" }));
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    expect(screen.queryByText("Shohei Ohtani")).not.toBeInTheDocument();
  });
});
