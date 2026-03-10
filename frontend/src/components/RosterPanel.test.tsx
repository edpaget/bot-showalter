import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { DraftPickType, RosterSlotType } from "../generated/graphql";
import { RosterPanel } from "./RosterPanel";

const ROSTER: DraftPickType[] = [
  { pickNumber: 1, team: 1, playerId: 100, playerName: "Mike Trout", position: "OF", price: null },
  { pickNumber: 5, team: 1, playerId: 200, playerName: "Gerrit Cole", position: "SP", price: null },
];

const NEEDS: RosterSlotType[] = [
  { position: "C", remaining: 1 },
  { position: "OF", remaining: 2 },
];

describe("RosterPanel", () => {
  afterEach(cleanup);

  it("renders roster players", () => {
    render(<RosterPanel roster={ROSTER} needs={NEEDS} budgetRemaining={null} format="snake" />);
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    expect(screen.getByText("Gerrit Cole")).toBeInTheDocument();
  });

  it("shows empty slots for needs", () => {
    render(<RosterPanel roster={[]} needs={NEEDS} budgetRemaining={null} format="snake" />);
    const empties = screen.getAllByText("empty");
    expect(empties).toHaveLength(3); // 1 C + 2 OF
  });

  it("shows player count", () => {
    render(<RosterPanel roster={ROSTER} needs={NEEDS} budgetRemaining={null} format="snake" />);
    expect(screen.getByText("Players: 2")).toBeInTheDocument();
  });

  it("shows budget in auction format", () => {
    const auctionRoster: DraftPickType[] = [
      { pickNumber: 1, team: 1, playerId: 100, playerName: "Mike Trout", position: "OF", price: 35 },
    ];
    render(<RosterPanel roster={auctionRoster} needs={NEEDS} budgetRemaining={225} format="auction" />);
    expect(screen.getByText("Budget: $225")).toBeInTheDocument();
    expect(screen.getByText("Spent: $35")).toBeInTheDocument();
  });

  it("hides budget info in snake format", () => {
    render(<RosterPanel roster={ROSTER} needs={NEEDS} budgetRemaining={null} format="snake" />);
    expect(screen.queryByText(/Budget/)).not.toBeInTheDocument();
  });

  it("shows price next to player name in auction", () => {
    const auctionRoster: DraftPickType[] = [
      { pickNumber: 1, team: 1, playerId: 100, playerName: "Mike Trout", position: "OF", price: 35 },
    ];
    render(<RosterPanel roster={auctionRoster} needs={[]} budgetRemaining={225} format="auction" />);
    expect(screen.getByText("Mike Trout ($35)")).toBeInTheDocument();
  });
});
