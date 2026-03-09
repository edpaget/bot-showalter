import type { RosterSlot } from "../types/session";
import { displayPosition } from "../types/position";

interface NeedsPanelProps {
  needs: RosterSlot[];
}

export function NeedsPanel({ needs }: NeedsPanelProps) {
  const totalRemaining = needs.reduce((sum, n) => sum + n.remaining, 0);

  return (
    <div className="border border-gray-200 rounded p-3">
      <h3 className="text-sm font-semibold mb-2">Needs</h3>
      {needs.length === 0 ? (
        <p className="text-xs text-gray-500">Roster complete</p>
      ) : (
        <>
          <div className="flex flex-wrap gap-2">
            {needs.map((need) => (
              <div
                key={need.position}
                className="flex items-center gap-1 bg-amber-50 border border-amber-200 rounded px-2 py-1 text-xs"
              >
                <span className="font-medium">{displayPosition(need.position)}</span>
                <span className="text-gray-500">×{need.remaining}</span>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">{totalRemaining} slots remaining</p>
        </>
      )}
    </div>
  );
}
