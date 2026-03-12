import { useQuery } from "@apollo/client";
import type { WebConfigQuery, YahooStandingsQuery, YahooTeamsQuery } from "../generated/graphql";
import { WEB_CONFIG_QUERY, YAHOO_STANDINGS_QUERY, YAHOO_TEAMS_QUERY } from "../graphql/queries";

export function LeagueView() {
  const { data: configData, loading: configLoading } = useQuery<WebConfigQuery>(WEB_CONFIG_QUERY);
  const yahooLeague = configData?.webConfig?.yahooLeague;

  const { data: teamsData, loading: teamsLoading } = useQuery<YahooTeamsQuery>(YAHOO_TEAMS_QUERY, {
    variables: { leagueKey: yahooLeague?.leagueKey ?? "" },
    skip: !yahooLeague,
  });

  const { data: standingsData, loading: standingsLoading } = useQuery<YahooStandingsQuery>(YAHOO_STANDINGS_QUERY, {
    variables: {
      leagueKey: yahooLeague?.leagueKey ?? "",
      season: yahooLeague?.season ?? 0,
    },
    skip: !yahooLeague,
  });

  if (configLoading) {
    return <div className="p-6 text-gray-400">Loading...</div>;
  }

  if (!yahooLeague) {
    return (
      <div className="p-6 text-gray-400">
        No Yahoo league configured. Start the server with <code>--yahoo-config-dir</code> to enable league features.
      </div>
    );
  }

  const teams = teamsData?.yahooTeams ?? [];
  const standings = standingsData?.yahooStandings ?? [];

  // Build a lookup from team_key to team info for manager names and ownership
  const teamLookup = new Map(teams.map((t) => [t.teamKey, t]));

  // Extract stat column names from the first standings entry
  const firstEntry = standings[0];
  const statColumns = firstEntry ? Object.keys(firstEntry.statValues as Record<string, number>) : [];

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold mb-1">{yahooLeague.leagueName}</h1>
      <p className="text-gray-400 mb-6">
        {yahooLeague.season} &middot; {yahooLeague.numTeams} teams
        {yahooLeague.isKeeper && " \u00b7 Keeper"}
      </p>

      {teamsLoading || standingsLoading ? (
        <div className="text-gray-400">Loading standings...</div>
      ) : standings.length === 0 ? (
        <div className="text-gray-400">
          No standings data available. Run <code>yahoo standings</code> to sync.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-gray-700 text-left text-gray-400">
                <th className="py-2 pr-4">Rank</th>
                <th className="py-2 pr-4">Team</th>
                <th className="py-2 pr-4">Manager</th>
                {statColumns.map((col) => (
                  <th key={col} className="py-2 pr-4 text-right">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {standings.map((entry) => {
                const team = teamLookup.get(entry.teamKey);
                const isUser = team?.isOwnedByUser ?? false;
                const stats = entry.statValues as Record<string, number>;
                return (
                  <tr
                    key={entry.teamKey}
                    className={`border-b border-gray-800 ${isUser ? "bg-blue-900/30 font-semibold" : ""}`}
                  >
                    <td className="py-2 pr-4">{entry.finalRank}</td>
                    <td className="py-2 pr-4">{entry.teamName}</td>
                    <td className="py-2 pr-4 text-gray-400">{team?.managerName ?? ""}</td>
                    {statColumns.map((col) => (
                      <td key={col} className="py-2 pr-4 text-right tabular-nums">
                        {typeof stats[col] === "number" ? stats[col].toLocaleString() : stats[col]}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
