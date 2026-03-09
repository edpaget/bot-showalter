import { render, screen, cleanup } from "@testing-library/react";
import { describe, it, expect, afterEach } from "vitest";
import { NeedsPanel } from "./NeedsPanel";
import type { RosterSlot } from "../types/session";

const NEEDS: RosterSlot[] = [
  { position: "C", remaining: 1 },
  { position: "OF", remaining: 3 },
  { position: "SP", remaining: 2 },
];

describe("NeedsPanel", () => {
  afterEach(cleanup);

  it("renders all unfilled positions", () => {
    render(<NeedsPanel needs={NEEDS} />);
    expect(screen.getByText("C")).toBeInTheDocument();
    expect(screen.getByText("OF")).toBeInTheDocument();
    expect(screen.getByText("SP")).toBeInTheDocument();
  });

  it("shows remaining count per position", () => {
    render(<NeedsPanel needs={NEEDS} />);
    expect(screen.getByText("×3")).toBeInTheDocument();
    expect(screen.getByText("×2")).toBeInTheDocument();
  });

  it("shows total remaining slots", () => {
    render(<NeedsPanel needs={NEEDS} />);
    expect(screen.getByText("6 slots remaining")).toBeInTheDocument();
  });

  it("shows complete message when no needs", () => {
    render(<NeedsPanel needs={[]} />);
    expect(screen.getByText("Roster complete")).toBeInTheDocument();
  });
});
