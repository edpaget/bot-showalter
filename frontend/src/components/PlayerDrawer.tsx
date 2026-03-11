import { useQuery } from "@apollo/client";
import { usePlayerDrawer } from "../context/PlayerDrawerContext";
import type {
  CategoryConfigType,
  LeagueQuery,
  PlayerBioQuery,
  ProjectionsQuery,
  ValuationsQuery,
} from "../generated/graphql";
import { LEAGUE_QUERY, PLAYER_BIO_QUERY, PROJECTIONS_QUERY, VALUATIONS_QUERY } from "../graphql/queries";
import { displayPosition } from "../lib/position";

export function PlayerDrawer() {
  const { isOpen, playerId, playerName, playerType: ctxPlayerType, season, closeDrawer } = usePlayerDrawer();

  const { data: bioData, loading: bioLoading } = useQuery<PlayerBioQuery>(PLAYER_BIO_QUERY, {
    variables: { playerId, season },
    skip: !isOpen || playerId == null,
  });

  const { data: projData, loading: projLoading } = useQuery<ProjectionsQuery>(PROJECTIONS_QUERY, {
    variables: { season, playerName: playerName ?? "" },
    skip: !isOpen || !playerName,
  });

  const { data: valData, loading: valLoading } = useQuery<ValuationsQuery>(VALUATIONS_QUERY, {
    variables: { season, top: 500 },
    skip: !isOpen,
  });

  const { data: leagueData } = useQuery<LeagueQuery>(LEAGUE_QUERY, {
    skip: !isOpen,
  });

  if (!isOpen) return null;

  const bio = bioData?.playerBio;
  const allProjections = projData?.projections ?? [];
  // Use playerType from context (passed by draft board), fall back to bio position
  const playerType = ctxPlayerType
    ?? (bio ? (["SP", "RP", "P"].includes(bio.primaryPosition) ? "pitcher" : "batter") : null);
  const projections = playerType ? allProjections.filter((p) => p.playerType === playerType) : allProjections;
  const allValuations = valData?.valuations ?? [];
  const valuations = playerName ? allValuations.filter((v) => v.playerName === playerName) : [];
  const leagueCategories: CategoryConfigType[] =
    playerType === "pitcher"
      ? (leagueData?.league.pitchingCategories ?? [])
      : (leagueData?.league.battingCategories ?? []);

  return (
    <>
      {/* biome-ignore lint/a11y/useKeyWithClickEvents: backdrop dismiss doesn't need keyboard handler */}
      {/* biome-ignore lint/a11y/useSemanticElements: full-screen backdrop overlay isn't a semantic button */}
      <div
        role="button"
        tabIndex={-1}
        className="fixed inset-0 bg-black/30 z-40"
        onClick={closeDrawer}
        data-testid="drawer-backdrop"
      />
      <div className="fixed right-0 top-0 h-full w-[400px] bg-white shadow-lg z-50 overflow-auto p-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold">{playerName}</h2>
          <button
            type="button"
            onClick={closeDrawer}
            className="text-gray-500 hover:text-gray-800 text-xl"
            aria-label="Close drawer"
          >
            ×
          </button>
        </div>

        {/* Bio Section */}
        <section className="mb-4">
          <h3 className="font-semibold text-sm text-gray-600 mb-2">Biography</h3>
          {bioLoading ? (
            <p className="text-gray-400 text-sm">Loading...</p>
          ) : bio ? (
            <div className="grid grid-cols-2 gap-1 text-sm">
              <span className="text-gray-500">Team</span>
              <span>{bio.team}</span>
              <span className="text-gray-500">Age</span>
              <span>{bio.age ?? "—"}</span>
              <span className="text-gray-500">Position</span>
              <span>{bio.primaryPosition}</span>
              <span className="text-gray-500">Bats/Throws</span>
              <span>
                {bio.bats ?? "—"}/{bio.throws ?? "—"}
              </span>
              <span className="text-gray-500">Experience</span>
              <span>{bio.experience} yr</span>
            </div>
          ) : (
            <p className="text-gray-400 text-sm">No data available</p>
          )}
        </section>

        {/* Projections Section */}
        <section className="mb-4">
          <h3 className="font-semibold text-sm text-gray-600 mb-2">Projections</h3>
          {projLoading || bioLoading ? (
            <p className="text-gray-400 text-sm">Loading...</p>
          ) : projections.length > 0 ? (
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr>
                  <th className="border border-gray-200 px-2 py-1 bg-gray-50 text-left">System</th>
                  {leagueCategories.map((cat) => (
                    <th key={cat.key} className="border border-gray-200 px-2 py-1 bg-gray-50 text-right">
                      {cat.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {projections.map((p, i) => (
                  <tr key={i}>
                    <td className="border border-gray-200 px-2 py-1">{p.system}</td>
                    {leagueCategories.map((cat) => (
                      <td key={cat.key} className="border border-gray-200 px-2 py-1 text-right font-mono">
                        {formatStat(p.stats[cat.key] as number | undefined, cat.statType)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-gray-400 text-sm">No projections</p>
          )}
        </section>

        {/* Valuations Section */}
        <section>
          <h3 className="font-semibold text-sm text-gray-600 mb-2">Valuations</h3>
          {valLoading ? (
            <p className="text-gray-400 text-sm">Loading...</p>
          ) : valuations.length > 0 ? (
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr>
                  <th className="border border-gray-200 px-2 py-1 bg-gray-50 text-left">System</th>
                  <th className="border border-gray-200 px-2 py-1 bg-gray-50 text-left">Version</th>
                  <th className="border border-gray-200 px-2 py-1 bg-gray-50 text-left">Pos</th>
                  <th className="border border-gray-200 px-2 py-1 bg-gray-50 text-left">Value</th>
                  <th className="border border-gray-200 px-2 py-1 bg-gray-50 text-left">Rank</th>
                </tr>
              </thead>
              <tbody>
                {valuations.map((v, i) => (
                  <tr key={i}>
                    <td className="border border-gray-200 px-2 py-1">{v.system}</td>
                    <td className="border border-gray-200 px-2 py-1">{v.version}</td>
                    <td className="border border-gray-200 px-2 py-1">{displayPosition(v.position)}</td>
                    <td className="border border-gray-200 px-2 py-1 font-mono">${v.value.toFixed(1)}</td>
                    <td className="border border-gray-200 px-2 py-1">{v.rank}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-gray-400 text-sm">No valuations</p>
          )}
        </section>
      </div>
    </>
  );
}

function formatStat(value: number | undefined, statType: string): string {
  if (value == null) return "—";
  if (statType === "rate") return value.toFixed(3);
  return Math.round(value).toString();
}
