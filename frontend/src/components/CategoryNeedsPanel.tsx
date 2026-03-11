import type { CategoryNeedType } from "../generated/graphql";

interface CategoryNeedsPanelProps {
  needs: CategoryNeedType[];
  onPlayerClick?: (playerId: number, playerType: string) => void;
}

export function CategoryNeedsPanel({ needs, onPlayerClick }: CategoryNeedsPanelProps) {
  if (needs.length === 0) {
    return (
      <div className="border border-gray-200 rounded p-3">
        <h3 className="text-sm font-semibold mb-2">Category Needs</h3>
        <p className="text-xs text-gray-500">No weak categories</p>
      </div>
    );
  }

  return (
    <div className="border border-gray-200 rounded p-3">
      <h3 className="text-sm font-semibold mb-2">Category Needs</h3>
      <div className="space-y-3">
        {needs.map((need) => (
          <div key={need.category}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium uppercase text-red-600">{need.category}</span>
              <span className="text-xs text-gray-500">
                #{need.currentRank} → #{need.targetRank}
              </span>
            </div>
            <div className="space-y-0.5">
              {need.bestAvailable.map((player) => (
                <div key={player.playerId} className="flex items-center justify-between text-xs">
                  <button
                    type="button"
                    className="text-blue-600 hover:underline truncate text-left"
                    onClick={() => onPlayerClick?.(player.playerId, "batter")}
                  >
                    {player.playerName}
                  </button>
                  <div className="flex items-center gap-1.5 shrink-0 ml-2">
                    <span className="text-green-600">+{player.categoryImpact.toFixed(1)}</span>
                    {player.tradeoffCategories.length > 0 && (
                      <span className="text-orange-500" title={`Hurts: ${player.tradeoffCategories.join(", ")}`}>
                        ⚠
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
