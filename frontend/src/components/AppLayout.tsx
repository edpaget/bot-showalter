import { useQuery } from "@apollo/client";
import { NavLink, Outlet } from "react-router-dom";
import { WEB_CONFIG_QUERY } from "../graphql/queries";
import { LeagueBadge } from "./LeagueBadge";
import { PlayerDrawer } from "./PlayerDrawer";

const NAV_ITEMS = [
  { to: "/", label: "Draft" },
  { to: "/projections", label: "Projections" },
  { to: "/valuations", label: "Valuations" },
  { to: "/adp", label: "ADP Report" },
  { to: "/players", label: "Player Search" },
  { to: "/keeper-planner", label: "Keeper Planner" },
];

export function AppLayout() {
  const { data } = useQuery(WEB_CONFIG_QUERY);
  const yahooLeague = data?.webConfig?.yahooLeague ?? null;

  return (
    <div className="flex flex-col h-screen">
      <nav className="bg-gray-800 text-white px-4 py-2 flex gap-4 items-center flex-shrink-0">
        <span className="font-bold mr-4">FBM</span>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              `px-3 py-1 rounded text-sm ${
                isActive ? "bg-blue-600 text-white" : "text-gray-300 hover:text-white hover:bg-gray-700"
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
        {yahooLeague && (
          <div className="ml-auto">
            <LeagueBadge leagueName={yahooLeague.leagueName} season={yahooLeague.season} />
          </div>
        )}
      </nav>
      <div className="flex-1 min-h-0 overflow-auto">
        <Outlet />
      </div>
      <PlayerDrawer />
    </div>
  );
}
