import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import type { DraftPick } from "../types/session";
import { PickLogPanel } from "./PickLogPanel";

const PICKS: DraftPick[] = [
  { pickNumber: 1, team: 1, playerId: 100, playerName: "Mike Trout", position: "OF", price: 35 },
  { pickNumber: 2, team: 2, playerId: 200, playerName: "Gerrit Cole", position: "SP", price: 28 },
  { pickNumber: 3, team: 1, playerId: 300, playerName: "Aaron Judge", position: "OF", price: 32 },
];

describe("PickLogPanel", () => {
  afterEach(cleanup);

  it("renders all picks", () => {
    render(<PickLogPanel picks={PICKS} />);
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    expect(screen.getByText("Gerrit Cole")).toBeInTheDocument();
    expect(screen.getByText("Aaron Judge")).toBeInTheDocument();
  });

  it("shows pick count in header", () => {
    render(<PickLogPanel picks={PICKS} />);
    expect(screen.getByText("Pick Log (3 picks)")).toBeInTheDocument();
  });

  it("highlights most recent pick", () => {
    render(<PickLogPanel picks={PICKS} />);
    const judgeRow = screen.getByText("Aaron Judge").closest("tr")!;
    expect(judgeRow.className).toContain("bg-yellow-50");
  });

  it("collapses when header is clicked", async () => {
    render(<PickLogPanel picks={PICKS} />);
    const user = userEvent.setup();
    await user.click(screen.getByText(/Pick Log/));
    expect(screen.queryByText("Mike Trout")).not.toBeInTheDocument();
  });

  it("shows empty message when no picks", () => {
    render(<PickLogPanel picks={[]} />);
    expect(screen.getByText("No picks yet")).toBeInTheDocument();
  });

  it("shows prices", () => {
    render(<PickLogPanel picks={PICKS} />);
    expect(screen.getByText("$35")).toBeInTheDocument();
  });

  it("shows most recent picks first", () => {
    render(<PickLogPanel picks={PICKS} />);
    const rows = screen.getAllByRole("row").slice(1); // skip header
    const firstPlayerCell = rows[0]!.querySelectorAll("td")[2]!;
    expect(firstPlayerCell.textContent).toBe("Aaron Judge");
  });
});
