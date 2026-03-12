import { describe, expect, it } from "vitest";
import type { DraftTradeType } from "../generated/graphql";
import { remainingPicksForTeam, snakeTeam, teamForPick } from "./snakePicks";

describe("snakeTeam", () => {
  it("returns correct teams for round 1 (even round, forward)", () => {
    expect(snakeTeam(1, 4)).toBe(1);
    expect(snakeTeam(2, 4)).toBe(2);
    expect(snakeTeam(3, 4)).toBe(3);
    expect(snakeTeam(4, 4)).toBe(4);
  });

  it("returns correct teams for round 2 (odd round, reverse)", () => {
    expect(snakeTeam(5, 4)).toBe(4);
    expect(snakeTeam(6, 4)).toBe(3);
    expect(snakeTeam(7, 4)).toBe(2);
    expect(snakeTeam(8, 4)).toBe(1);
  });

  it("returns correct teams for round 3 (even round, forward again)", () => {
    expect(snakeTeam(9, 4)).toBe(1);
    expect(snakeTeam(10, 4)).toBe(2);
    expect(snakeTeam(11, 4)).toBe(3);
    expect(snakeTeam(12, 4)).toBe(4);
  });
});

describe("teamForPick", () => {
  const noTrades: DraftTradeType[] = [];

  it("falls back to snakeTeam when no trades", () => {
    expect(teamForPick(1, 4, noTrades)).toBe(1);
    expect(teamForPick(5, 4, noTrades)).toBe(4);
  });

  it("applies trade overrides", () => {
    const trades: DraftTradeType[] = [{ teamA: 1, teamB: 3, teamAGives: [9], teamBGives: [11] }];
    // Pick 9 was team 1's, now goes to team 3
    expect(teamForPick(9, 4, trades)).toBe(3);
    // Pick 11 was team 3's, now goes to team 1
    expect(teamForPick(11, 4, trades)).toBe(1);
    // Unaffected picks unchanged
    expect(teamForPick(10, 4, trades)).toBe(2);
  });

  it("handles multiple trades", () => {
    const trades: DraftTradeType[] = [
      { teamA: 1, teamB: 2, teamAGives: [1], teamBGives: [2] },
      { teamA: 3, teamB: 4, teamAGives: [3], teamBGives: [4] },
    ];
    expect(teamForPick(1, 4, trades)).toBe(2);
    expect(teamForPick(2, 4, trades)).toBe(1);
    expect(teamForPick(3, 4, trades)).toBe(4);
    expect(teamForPick(4, 4, trades)).toBe(3);
  });
});

describe("remainingPicksForTeam", () => {
  it("returns remaining picks for a team with no trades", () => {
    // 4 teams, 8 picks total. Team 1 has picks 1 and 8.
    const result = remainingPicksForTeam(1, 1, 8, 4, []);
    expect(result).toEqual([1, 8]);
  });

  it("respects currentPick filter", () => {
    // Team 1 has picks 1 and 8, but current is 5 so only 8 remains
    const result = remainingPicksForTeam(1, 5, 8, 4, []);
    expect(result).toEqual([8]);
  });

  it("includes traded-in picks and excludes traded-out picks", () => {
    const trades: DraftTradeType[] = [{ teamA: 1, teamB: 2, teamAGives: [8], teamBGives: [7] }];
    // Team 1 originally has picks 1,8. Gave away 8, got 7.
    const result = remainingPicksForTeam(1, 1, 8, 4, trades);
    expect(result).toEqual([1, 7]);
  });
});
