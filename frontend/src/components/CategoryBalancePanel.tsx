import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer } from "recharts";
import type { CategoryBalanceType } from "../generated/graphql";

interface CategoryBalancePanelProps {
  balance: CategoryBalanceType[];
}

const STRENGTH_COLORS: Record<string, string> = {
  elite: "text-green-700 bg-green-50",
  strong: "text-green-600 bg-green-50",
  average: "text-gray-600 bg-gray-50",
  weak: "text-orange-600 bg-orange-50",
  poor: "text-red-600 bg-red-50",
};

export function CategoryBalancePanel({ balance }: CategoryBalancePanelProps) {
  if (balance.length === 0) {
    return (
      <div className="border border-gray-200 rounded p-3">
        <h3 className="text-sm font-semibold mb-2">Category Balance</h3>
        <p className="text-xs text-gray-500">Draft players to see balance</p>
      </div>
    );
  }

  const chartData = balance.map((b) => ({
    category: b.category,
    value: b.projectedValue,
    rank: b.leagueRankEstimate,
  }));

  return (
    <div className="border border-gray-200 rounded p-3">
      <h3 className="text-sm font-semibold mb-2">Category Balance</h3>
      <ResponsiveContainer width="100%" height={200}>
        <RadarChart data={chartData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="category" tick={{ fontSize: 10 }} />
          <PolarRadiusAxis tick={false} axisLine={false} />
          <Radar dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
        </RadarChart>
      </ResponsiveContainer>
      <table className="w-full text-xs mt-2">
        <thead>
          <tr className="text-left text-gray-500">
            <th className="pb-1">Category</th>
            <th className="pb-1">Rank</th>
            <th className="pb-1">Strength</th>
          </tr>
        </thead>
        <tbody>
          {balance.map((b) => (
            <tr key={b.category} className="border-t border-gray-100">
              <td className="py-0.5">{b.category}</td>
              <td className="py-0.5">#{b.leagueRankEstimate}</td>
              <td className="py-0.5">
                <span
                  className={`px-1.5 py-0.5 rounded text-xs ${STRENGTH_COLORS[b.strength] ?? "text-gray-600 bg-gray-50"}`}
                >
                  {b.strength}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
