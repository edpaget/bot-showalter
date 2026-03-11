import { useLazyQuery, useMutation } from "@apollo/client";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { PlanKeeperDraftQuery } from "../generated/graphql";

type Scenario = NonNullable<PlanKeeperDraftQuery["planKeeperDraft"]>["scenarios"][number];

import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import { START_SESSION } from "../graphql/mutations";
import { PLAN_KEEPER_DRAFT_QUERY } from "../graphql/queries";

export function KeeperPlannerView() {
  const [season, setSeason] = useState(2026);
  const [maxKeepers, setMaxKeepers] = useState(5);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const navigate = useNavigate();
  const { openPlayer } = usePlayerDrawer();

  const [loadScenarios, { data, loading, error }] = useLazyQuery<PlanKeeperDraftQuery>(PLAN_KEEPER_DRAFT_QUERY);

  const [startSession] = useMutation(START_SESSION);

  const scenarios = data?.planKeeperDraft?.scenarios ?? [];
  const selected: Scenario | undefined = scenarios[selectedIndex];

  function handleLoad() {
    setSelectedIndex(0);
    loadScenarios({
      variables: { season, maxKeepers, boardPreviewSize: 20 },
    });
  }

  async function handleStartDraft() {
    if (!selected) return;
    const result = await startSession({
      variables: {
        season,
        system: null,
        version: null,
        teams: null,
        budget: null,
        keeperPlayerIds: selected.keeperIds,
      },
    });
    if (result.data?.startSession?.sessionId) {
      navigate("/");
    }
  }

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-2xl font-bold">Keeper Planner</h1>

      {/* Config bar */}
      <div className="flex gap-4 items-end">
        <div>
          <label className="block text-sm text-gray-500">Season</label>
          <input
            type="number"
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
            className="border rounded px-2 py-1 w-24"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-500">Max Keepers</label>
          <input
            type="number"
            value={maxKeepers}
            onChange={(e) => setMaxKeepers(Number(e.target.value))}
            className="border rounded px-2 py-1 w-24"
            min={1}
          />
        </div>
        <button
          type="button"
          onClick={handleLoad}
          className="bg-blue-600 text-white px-4 py-1 rounded hover:bg-blue-700"
        >
          Load Scenarios
        </button>
      </div>

      {loading && <p className="text-gray-500">Loading scenarios...</p>}
      {error && <p className="text-red-600">Error: {error.message}</p>}

      {scenarios.length > 0 && (
        <div className="grid grid-cols-12 gap-4">
          {/* Scenario cards */}
          <div className="col-span-3 space-y-2">
            <h2 className="font-semibold text-lg">Scenarios</h2>
            {scenarios.map((s, i) => (
              <button
                type="button"
                key={s.keeperIds.join(",")}
                onClick={() => setSelectedIndex(i)}
                className={`w-full text-left p-3 rounded border ${
                  i === selectedIndex ? "border-blue-600 bg-blue-50" : "border-gray-200 hover:border-gray-400"
                }`}
              >
                <div className="font-medium">{i === 0 ? "Optimal" : `Alternative ${i}`}</div>
                <div className="text-sm text-gray-600">
                  {s.keepers.length} keepers · ${s.totalSurplus.toFixed(1)} surplus
                </div>
                <div className="text-xs text-gray-400 mt-1">{s.keepers.map((k) => k.playerName).join(", ")}</div>
              </button>
            ))}
          </div>

          {/* Selected scenario detail */}
          {selected && (
            <div className="col-span-9 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-lg">Scenario Detail</h2>
                <button
                  type="button"
                  onClick={handleStartDraft}
                  className="bg-green-600 text-white px-4 py-1 rounded hover:bg-green-700"
                >
                  Start Draft with This Set
                </button>
              </div>

              {/* Keeper table */}
              <div>
                <h3 className="font-medium mb-1">Keepers</h3>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-gray-500">
                      <th className="py-1">Name</th>
                      <th>Pos</th>
                      <th className="text-right">Cost</th>
                      <th className="text-right">Value</th>
                      <th className="text-right">Surplus</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selected.keepers.map((k) => (
                      <tr key={k.playerId} className="border-b">
                        <td className="py-1">
                          <button
                            type="button"
                            className="cursor-pointer text-blue-600 hover:underline bg-transparent border-none p-0"
                            onClick={() => openPlayer(k.playerId, k.playerName)}
                          >
                            {k.playerName}
                          </button>
                        </td>
                        <td>{k.position}</td>
                        <td className="text-right">${k.cost.toFixed(0)}</td>
                        <td className="text-right">{k.projectedValue.toFixed(1)}</td>
                        <td className={`text-right ${k.surplus >= 0 ? "text-green-600" : "text-red-600"}`}>
                          {k.surplus >= 0 ? "+" : ""}
                          {k.surplus.toFixed(1)}
                        </td>
                        <td>
                          <span
                            className={`text-xs px-1 rounded ${k.recommendation === "keep" ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}
                          >
                            {k.recommendation}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Strengths / Weaknesses */}
              <div className="flex gap-6">
                {selected.strongestCategories.length > 0 && (
                  <div>
                    <h3 className="font-medium mb-1">Strengths</h3>
                    <div className="flex gap-1">
                      {selected.strongestCategories.map((c) => (
                        <span key={c} className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {selected.weakestCategories.length > 0 && (
                  <div>
                    <h3 className="font-medium mb-1">Weaknesses</h3>
                    <div className="flex gap-1">
                      {selected.weakestCategories.map((c) => (
                        <span key={c} className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded">
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Category needs */}
              {selected.categoryNeeds.length > 0 && (
                <div>
                  <h3 className="font-medium mb-1">Category Needs</h3>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left text-gray-500">
                        <th className="py-1">Category</th>
                        <th className="text-right">Current Rank</th>
                        <th className="text-right">Target Rank</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.categoryNeeds.map((n) => (
                        <tr key={n.category} className="border-b">
                          <td className="py-1">{n.category}</td>
                          <td className="text-right">{n.currentRank}</td>
                          <td className="text-right">{n.targetRank}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Board preview */}
              <div>
                <h3 className="font-medium mb-1">Top Available Players (Adjusted)</h3>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-gray-500">
                      <th className="py-1">Name</th>
                      <th>Type</th>
                      <th>Pos</th>
                      <th className="text-right">Adj. Value</th>
                      <th className="text-right">Orig. Value</th>
                      <th className="text-right">Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selected.boardPreview.map((p) => (
                      <tr key={p.playerId} className="border-b">
                        <td className="py-1">
                          <button
                            type="button"
                            className="cursor-pointer text-blue-600 hover:underline bg-transparent border-none p-0"
                            onClick={() => openPlayer(p.playerId, p.playerName)}
                          >
                            {p.playerName}
                          </button>
                        </td>
                        <td>{p.playerType}</td>
                        <td>{p.position}</td>
                        <td className="text-right">{p.adjustedValue.toFixed(1)}</td>
                        <td className="text-right">{p.originalValue.toFixed(1)}</td>
                        <td className={`text-right ${p.valueChange >= 0 ? "text-green-600" : "text-red-600"}`}>
                          {p.valueChange >= 0 ? "+" : ""}
                          {p.valueChange.toFixed(1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Scarcity */}
              {selected.scarcity.length > 0 && (
                <div>
                  <h3 className="font-medium mb-1">Position Scarcity</h3>
                  <div className="flex gap-3 flex-wrap">
                    {selected.scarcity.map((s) => (
                      <div key={s.position} className="border rounded p-2 text-sm">
                        <div className="font-medium">{s.position}</div>
                        <div className="text-gray-500">Tier 1: {s.tier1Value.toFixed(1)}</div>
                        <div className="text-gray-500">Repl: {s.replacementValue.toFixed(1)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {!loading && scenarios.length === 0 && data && (
        <p className="text-gray-500">No scenarios available. Make sure keeper costs are configured.</p>
      )}
    </div>
  );
}
