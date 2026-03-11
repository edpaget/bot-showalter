import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { LeagueBadge } from "./LeagueBadge";

describe("LeagueBadge", () => {
  afterEach(cleanup);

  it("renders league name and season", () => {
    render(<LeagueBadge leagueName="Test League" season={2026} />);
    expect(screen.getByText("Test League")).toBeInTheDocument();
    expect(screen.getByText("2026")).toBeInTheDocument();
  });
});
