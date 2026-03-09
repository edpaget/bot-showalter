import type { DraftPick, RosterSlot } from "../types/session";

interface RosterPanelProps {
  roster: DraftPick[];
  needs: RosterSlot[];
  budgetRemaining: number | null;
  format: string;
}

export function RosterPanel({ roster, needs, budgetRemaining, format }: RosterPanelProps) {
  const totalValue = roster.reduce((sum, p) => sum + (p.price ?? 0), 0);
  const allPositions = [
    ...needs.map((n) => n.position),
    ...roster.map((p) => p.position),
  ];
  const uniquePositions = [...new Set(allPositions)];

  return (
    <div className="border border-gray-200 rounded p-3">
      <h3 className="text-sm font-semibold mb-2">Roster</h3>
      <div className="space-y-1 text-xs">
        {uniquePositions.map((pos) => {
          const filled = roster.filter((p) => p.position === pos);
          const need = needs.find((n) => n.position === pos);
          const remaining = need?.remaining ?? 0;
          return (
            <div key={pos} className="flex items-center gap-2">
              <span className="font-medium w-8">{pos}</span>
              <div className="flex gap-1 flex-1 flex-wrap">
                {filled.map((p) => (
                  <span
                    key={p.pickNumber}
                    className="bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded text-xs"
                  >
                    {p.playerName}
                    {p.price != null && ` ($${p.price})`}
                  </span>
                ))}
                {Array.from({ length: remaining }, (_, i) => (
                  <span
                    key={`empty-${pos}-${i}`}
                    className="border border-dashed border-gray-300 text-gray-400 px-1.5 py-0.5 rounded text-xs"
                  >
                    empty
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-2 pt-2 border-t border-gray-100 text-xs text-gray-600 flex gap-4">
        <span>Players: {roster.length}</span>
        {format === "auction" && budgetRemaining != null && (
          <span>Budget: ${budgetRemaining}</span>
        )}
        {format === "auction" && <span>Spent: ${totalValue}</span>}
      </div>
    </div>
  );
}
