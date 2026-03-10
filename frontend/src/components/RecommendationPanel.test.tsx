import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RecommendationType } from "../generated/graphql";
import { RecommendationPanel } from "./RecommendationPanel";

const RECS: RecommendationType[] = [
  { playerId: 1, playerName: "Mike Trout", position: "OF", value: 35, score: 0.95, reason: "Best value" },
  { playerId: 2, playerName: "Gerrit Cole", position: "SP", value: 25, score: 0.85, reason: "Need SP" },
  { playerId: 3, playerName: "Pete Alonso", position: "FIRST_BASE", value: 20, score: 0.75, reason: "Need 1B" },
];

describe("RecommendationPanel", () => {
  afterEach(cleanup);

  it("renders all recommendations", () => {
    render(<RecommendationPanel recommendations={RECS} onDraft={vi.fn()} sessionActive />);
    expect(screen.getByText("Mike Trout")).toBeInTheDocument();
    expect(screen.getByText("Gerrit Cole")).toBeInTheDocument();
    expect(screen.getByText("Pete Alonso")).toBeInTheDocument();
  });

  it("filters by position", async () => {
    render(<RecommendationPanel recommendations={RECS} onDraft={vi.fn()} sessionActive />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "SP" }));
    expect(screen.getByText("Gerrit Cole")).toBeInTheDocument();
    expect(screen.queryByText("Mike Trout")).not.toBeInTheDocument();
  });

  it("calls onDraft when Draft button is clicked", async () => {
    const onDraft = vi.fn();
    render(<RecommendationPanel recommendations={RECS} onDraft={onDraft} sessionActive />);
    const user = userEvent.setup();
    const draftButtons = screen.getAllByRole("button", { name: "Draft" });
    await user.click(draftButtons[0]!);
    expect(onDraft).toHaveBeenCalledWith(1, "OF");
  });

  it("hides Draft buttons when session is not active", () => {
    render(<RecommendationPanel recommendations={RECS} onDraft={vi.fn()} sessionActive={false} />);
    expect(screen.queryByRole("button", { name: "Draft" })).not.toBeInTheDocument();
  });

  it("shows empty message when no recommendations", () => {
    render(<RecommendationPanel recommendations={[]} onDraft={vi.fn()} sessionActive />);
    expect(screen.getByText("No recommendations")).toBeInTheDocument();
  });

  it("shows reason as tooltip", () => {
    render(<RecommendationPanel recommendations={RECS} onDraft={vi.fn()} sessionActive />);
    expect(screen.getByText("Mike Trout").closest("td")).toHaveAttribute("title", "Best value");
  });
});
