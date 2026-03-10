import type { Position as PositionType } from "../generated/graphql";

/**
 * Runtime Position values for iteration (e.g., Object.values(Position)).
 * Type comes from codegen; this const provides the values at runtime.
 */
export const Position = {
  C: "C",
  FIRST_BASE: "FIRST_BASE",
  SECOND_BASE: "SECOND_BASE",
  THIRD_BASE: "THIRD_BASE",
  SS: "SS",
  LF: "LF",
  CF: "CF",
  RF: "RF",
  OF: "OF",
  DH: "DH",
  UTIL: "UTIL",
  SP: "SP",
  RP: "RP",
  P: "P",
} as const satisfies Record<string, PositionType>;

export type Position = PositionType;

const DISPLAY_LABELS: Record<Position, string> = {
  C: "C",
  FIRST_BASE: "1B",
  SECOND_BASE: "2B",
  THIRD_BASE: "3B",
  SS: "SS",
  LF: "LF",
  CF: "CF",
  RF: "RF",
  OF: "OF",
  DH: "DH",
  UTIL: "UTIL",
  BN: "BN",
  SP: "SP",
  RP: "RP",
  P: "P",
};

/** Convert a GraphQL Position enum value to a human-readable label (e.g., FIRST_BASE → "1B"). */
export function displayPosition(pos: string): string {
  return DISPLAY_LABELS[pos as Position] ?? pos;
}
