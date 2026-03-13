import type { DraftPickType, KeeperInfoType, RosterSlotType } from "../generated/graphql";
import { displayPosition } from "../lib/position";

interface RosterPanelProps {
  roster: DraftPickType[];
  keepers: KeeperInfoType[];
  needs: RosterSlotType[];
  budgetRemaining: number | null;
  format: string;
}

export function RosterPanel({ roster, keepers, needs, budgetRemaining, format }: RosterPanelProps) {
  const totalValue = roster.reduce((sum, p) => sum + (p.price ?? 0), 0);
  const keeperPositions = keepers.map((k) => k.position);
  const allPositions = [...needs.map((n) => n.position), ...roster.map((p) => p.position), ...keeperPositions];
  const uniquePositions = [...new Set(allPositions)];

  return (
    <div className="border border-gray-200 rounded p-3">
      <h3 className="text-sm font-semibold mb-2">Roster</h3>
      <div className="space-y-1 text-xs">
        {uniquePositions.map((pos) => {
          const filled = roster.filter((p) => p.position === pos);
          const kept = keepers.filter((k) => k.position === pos);
          const need = needs.find((n) => n.position === pos);
          const remaining = need?.remaining ?? 0;
          return (
            <div key={pos} className="flex items-center gap-2">
              <span className="font-medium w-8">{displayPosition(pos)}</span>
              <div className="flex gap-1 flex-1 flex-wrap">
                {kept.map((k) => (
                  <span key={`k-${k.playerId}`} className="bg-green-100 text-green-800 px-1.5 py-0.5 rounded text-xs">
                    {k.playerName}
                    {k.cost != null && ` ($${k.cost})`}
                  </span>
                ))}
                {filled.map((p) => (
                  <span key={p.pickNumber} className="bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded text-xs">
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
        <span>Players: {roster.length + keepers.length}</span>
        {format === "auction" && budgetRemaining != null && <span>Budget: ${budgetRemaining}</span>}
        {format === "auction" && <span>Spent: ${totalValue}</span>}
      </div>
    </div>
  );
}
