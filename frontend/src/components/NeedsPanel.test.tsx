import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { RosterSlotType } from "../generated/graphql";
import { NeedsPanel } from "./NeedsPanel";

const NEEDS: RosterSlotType[] = [
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
