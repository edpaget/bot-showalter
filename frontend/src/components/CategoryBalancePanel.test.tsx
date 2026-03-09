import { render, screen, cleanup } from "@testing-library/react";
import { describe, it, expect, afterEach, vi } from "vitest";
import { CategoryBalancePanel } from "./CategoryBalancePanel";
import type { CategoryBalance } from "../types/session";

// Mock recharts to avoid canvas issues in jsdom
vi.mock("recharts", () => ({
  RadarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="radar-chart">{children}</div>,
  PolarGrid: () => <div />,
  PolarAngleAxis: () => <div />,
  PolarRadiusAxis: () => <div />,
  Radar: () => <div />,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

const BALANCE: CategoryBalance[] = [
  { category: "HR", projectedValue: 245, leagueRankEstimate: 3, strength: "strong" },
  { category: "RBI", projectedValue: 850, leagueRankEstimate: 5, strength: "average" },
  { category: "W", projectedValue: 75, leagueRankEstimate: 10, strength: "weak" },
];

describe("CategoryBalancePanel", () => {
  afterEach(cleanup);

  it("renders category table with ranks and strengths", () => {
    render(<CategoryBalancePanel balance={BALANCE} />);
    expect(screen.getByText("HR")).toBeInTheDocument();
    expect(screen.getByText("#3")).toBeInTheDocument();
    expect(screen.getByText("strong")).toBeInTheDocument();
    expect(screen.getByText("weak")).toBeInTheDocument();
  });

  it("renders radar chart", () => {
    render(<CategoryBalancePanel balance={BALANCE} />);
    expect(screen.getByTestId("radar-chart")).toBeInTheDocument();
  });

  it("shows empty message when no balance data", () => {
    render(<CategoryBalancePanel balance={[]} />);
    expect(screen.getByText("Draft players to see balance")).toBeInTheDocument();
  });

  it("applies strength color classes", () => {
    render(<CategoryBalancePanel balance={BALANCE} />);
    const strongEl = screen.getByText("strong");
    expect(strongEl.className).toContain("text-green-600");
    const weakEl = screen.getByText("weak");
    expect(weakEl.className).toContain("text-orange-600");
  });
});
