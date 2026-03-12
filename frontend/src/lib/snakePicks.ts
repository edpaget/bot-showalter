import type { DraftTradeType } from "../generated/graphql";

export function snakeTeam(pickNumber: number, teams: number): number {
  const zero = pickNumber - 1;
  const round = Math.floor(zero / teams);
  const pos = zero % teams;
  return round % 2 === 0 ? pos + 1 : teams - pos;
}

export function teamForPick(pickNumber: number, teams: number, trades: DraftTradeType[]): number {
  const overrides = new Map<number, number>();
  for (const trade of trades) {
    for (const pick of trade.teamAGives) {
      overrides.set(pick, trade.teamB);
    }
    for (const pick of trade.teamBGives) {
      overrides.set(pick, trade.teamA);
    }
  }
  return overrides.get(pickNumber) ?? snakeTeam(pickNumber, teams);
}

export function remainingPicksForTeam(
  team: number,
  currentPick: number,
  totalPicks: number,
  teams: number,
  trades: DraftTradeType[],
): number[] {
  const result: number[] = [];
  for (let pick = currentPick; pick <= totalPicks; pick++) {
    if (teamForPick(pick, teams, trades) === team) {
      result.push(pick);
    }
  }
  return result;
}
