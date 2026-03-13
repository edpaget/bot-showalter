import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import type { DraftPickType } from "../generated/graphql";
import { PickLogPanel } from "./PickLogPanel";

const PICKS: DraftPickType[] = [
  { pickNumber: 1, team: 1, playerId: 100, playerName: "Mike Trout", position: "OF", playerType: "B", price: 35 },
  { pickNumber: 2, team: 2, playerId: 200, playerName: "Gerrit Cole", position: "SP", playerType: "P", price: 28 },
  { pickNumber: 3, team: 1, playerId: 300, playerName: "Aaron Judge", position: "OF", playerType: "B", price: 32 },
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

  it("displays team names when provided", () => {
    const teamNames = { 1: "Sluggers", 2: "Aces" };
    render(<PickLogPanel picks={PICKS} teamNames={teamNames} />);
    expect(screen.getAllByText("Sluggers")).toHaveLength(2); // picks 1 and 3
    expect(screen.getByText("Aces")).toBeInTheDocument();
  });

  it("falls back to Team N when no team names provided", () => {
    render(<PickLogPanel picks={PICKS} />);
    expect(screen.getAllByText(/^Team \d+$/)).toHaveLength(3);
  });

  it("displays trade team names when provided", () => {
    const trades = [{ teamA: 1, teamB: 2, teamAGives: [1], teamBGives: [2] }];
    const teamNames = { 1: "Sluggers", 2: "Aces" };
    render(<PickLogPanel picks={[]} trades={trades} teamNames={teamNames} userTeam={1} />);
    expect(screen.getByText(/Sluggers ↔ Aces/)).toBeInTheDocument();
  });
});
