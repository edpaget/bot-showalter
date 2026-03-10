import { useState } from "react";
import type { KeeperInfo } from "../context/DraftSessionContext";

interface KeeperPanelProps {
  keepers: KeeperInfo[];
  userTeamName?: string;
}

export function KeeperPanel({ keepers, userTeamName }: KeeperPanelProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  if (keepers.length === 0) return null;

  // Group by team
  const byTeam: Record<string, KeeperInfo[]> = {};
  for (const k of keepers) {
    const team = k.teamName;
    if (!byTeam[team]) byTeam[team] = [];
    byTeam[team].push(k);
  }
  const teamNames = Object.keys(byTeam).sort();

  const totalValue = keepers.reduce((sum, k) => sum + k.value, 0);

  const toggleTeam = (team: string) => {
    setCollapsed((prev) => ({ ...prev, [team]: !prev[team] }));
  };

  return (
    <div className="border border-gray-200 rounded p-3">
      <h3 className="text-sm font-semibold mb-2">Keepers</h3>
      <div className="text-xs text-gray-600 mb-2 flex gap-4">
        <span>Total: {keepers.length}</span>
        <span>Value: {totalValue.toFixed(1)}</span>
      </div>
      <div className="space-y-2">
        {teamNames.map((team) => {
          const isUser = userTeamName != null && team === userTeamName;
          const isCollapsed = collapsed[team] ?? false;
          const teamKeepers = byTeam[team] ?? [];
          return (
            <div
              key={team}
              className={`border rounded ${isUser ? "border-blue-300 bg-blue-50" : "border-gray-100"}`}
            >
              <button
                type="button"
                className="w-full text-left text-xs font-medium px-2 py-1 flex justify-between items-center"
                onClick={() => toggleTeam(team)}
              >
                <span>
                  {team} ({teamKeepers.length})
                  {isUser && <span className="ml-1 text-blue-600">(You)</span>}
                </span>
                <span className="text-gray-400">{isCollapsed ? "+" : "-"}</span>
              </button>
              {!isCollapsed && (
                <div className="px-2 pb-1 space-y-0.5">
                  {teamKeepers.map((k) => (
                    <div key={k.playerId} className="flex justify-between text-xs">
                      <span>
                        {k.playerName}{" "}
                        <span className="text-gray-400">{k.position}</span>
                      </span>
                      <span className="text-gray-500">
                        {k.cost != null && `$${k.cost}`}
                        {k.cost != null && " / "}
                        {k.value.toFixed(1)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
