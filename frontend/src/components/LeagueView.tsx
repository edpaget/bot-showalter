import { useQuery } from "@apollo/client";
import { useState } from "react";
import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import type { WebConfigQuery, YahooRostersQuery, YahooStandingsQuery, YahooTeamsQuery } from "../generated/graphql";
import { WEB_CONFIG_QUERY, YAHOO_ROSTERS_QUERY, YAHOO_STANDINGS_QUERY, YAHOO_TEAMS_QUERY } from "../graphql/queries";

export function LeagueView() {
  const [selectedTeamKey, setSelectedTeamKey] = useState<string | null>(null);
  const { openPlayer } = usePlayerDrawer();

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

  const { data: rostersData } = useQuery<YahooRostersQuery>(YAHOO_ROSTERS_QUERY, {
    variables: { leagueKey: yahooLeague?.leagueKey ?? "" },
    skip: !yahooLeague,
  });

  const rostersByTeam = new Map((rostersData?.yahooRosters ?? []).map((r) => [r.teamKey, r]));

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
                const isSelected = selectedTeamKey === entry.teamKey;
                const roster = rostersByTeam.get(entry.teamKey);
                return (
                  <>
                    <tr
                      key={entry.teamKey}
                      className={`border-b border-gray-800 cursor-pointer hover:bg-gray-800/50 ${isUser ? "bg-blue-900/30 font-semibold" : ""}`}
                      onClick={() => setSelectedTeamKey(isSelected ? null : entry.teamKey)}
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
                    {isSelected && roster && (
                      <tr key={`${entry.teamKey}-roster`} className="border-b border-gray-800 bg-gray-900/50">
                        <td colSpan={3 + statColumns.length} className="py-3 px-6">
                          <div className="text-xs text-gray-400 mb-2">
                            Roster (Week {roster.week}, {roster.asOf})
                          </div>
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="text-left text-gray-500 text-xs">
                                <th className="pb-1 pr-4">Player</th>
                                <th className="pb-1 pr-4">Position</th>
                                <th className="pb-1 pr-4">Acquired</th>
                              </tr>
                            </thead>
                            <tbody>
                              {roster.entries.map((e) => (
                                <tr key={e.yahooPlayerKey} className="border-t border-gray-800/50">
                                  <td className="py-1 pr-4">
                                    {e.playerId != null ? (
                                      <button
                                        type="button"
                                        className="text-blue-400 hover:underline"
                                        onClick={(ev) => {
                                          ev.stopPropagation();
                                          openPlayer(e.playerId as number, e.playerName);
                                        }}
                                      >
                                        {e.playerName}
                                      </button>
                                    ) : (
                                      <span className="text-gray-300">{e.playerName}</span>
                                    )}
                                  </td>
                                  <td className="py-1 pr-4 text-gray-400">{e.position}</td>
                                  <td className="py-1 pr-4 text-gray-400">{e.acquisitionType}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
