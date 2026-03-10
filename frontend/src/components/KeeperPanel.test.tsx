import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import type { KeeperInfo } from "../types/session";
import { KeeperPanel } from "./KeeperPanel";

const KEEPERS: KeeperInfo[] = [
  { playerId: 1, playerName: "Mike Trout", position: "OF", teamName: "Team A", cost: 35, value: 40.0 },
  { playerId: 2, playerName: "Shohei Ohtani", position: "OF", teamName: "Team A", cost: 30, value: 35.0 },
  { playerId: 3, playerName: "Gerrit Cole", position: "SP", teamName: "Team B", cost: 25, value: 30.0 },
];

describe("KeeperPanel", () => {
  afterEach(cleanup);

  it("renders grouped keepers by team", () => {
    render(<KeeperPanel keepers={KEEPERS} />);
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    expect(screen.getByText("Gerrit Cole")).toBeInTheDocument();
    expect(screen.getByText(/Team A \(2\)/)).toBeInTheDocument();
    expect(screen.getByText(/Team B \(1\)/)).toBeInTheDocument();
  });

  it("shows summary stats", () => {
    render(<KeeperPanel keepers={KEEPERS} />);
    expect(screen.getByText("Total: 3")).toBeInTheDocument();
    expect(screen.getByText("Value: 105.0")).toBeInTheDocument();
  });

  it("returns null when no keepers", () => {
    const { container } = render(<KeeperPanel keepers={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("highlights user team", () => {
    render(<KeeperPanel keepers={KEEPERS} userTeamName="Team A" />);
    expect(screen.getByText("(You)")).toBeInTheDocument();
  });

  it("collapses team on click", async () => {
    const user = userEvent.setup();
    render(<KeeperPanel keepers={KEEPERS} />);

    // Both players visible initially
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();

    // Click Team A header to collapse
    await user.click(screen.getByText(/Team A \(2\)/));
    expect(screen.queryByText("Mike Trout")).not.toBeInTheDocument();

    // Click again to expand
    await user.click(screen.getByText(/Team A \(2\)/));
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
  });

  it("shows cost and value for keepers", () => {
    render(<KeeperPanel keepers={KEEPERS} />);
    expect(screen.getByText("$35 / 40.0")).toBeInTheDocument();
  });

  it("shows only value when cost is null", () => {
    const noCosters: KeeperInfo[] = [
      { playerId: 1, playerName: "Mike Trout", position: "OF", teamName: "Team A", cost: null, value: 40.0 },
    ];
    render(<KeeperPanel keepers={noCosters} />);
    expect(screen.getByText("40.0")).toBeInTheDocument();
    expect(screen.queryByText(/\$/)).not.toBeInTheDocument();
  });
});
