import { BrowserRouter, Route, Routes } from "react-router-dom";
import { ADPReportView } from "./components/ADPReportView";
import { AppLayout } from "./components/AppLayout";
import { DraftDashboard } from "./components/DraftDashboard";
import { KeeperPlannerView } from "./components/KeeperPlannerView";
import { PlayerSearchView } from "./components/PlayerSearchView";
import { ProjectionsView } from "./components/ProjectionsView";
import { ValuationsView } from "./components/ValuationsView";
import { DraftSessionProvider } from "./context/DraftSessionContext";
import { PlayerDrawerProvider } from "./context/PlayerDrawerContext";

export default function App() {
  return (
    <BrowserRouter>
      <DraftSessionProvider>
        <PlayerDrawerProvider season={2026}>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/" element={<DraftDashboard season={2026} />} />
              <Route path="/projections" element={<ProjectionsView />} />
              <Route path="/valuations" element={<ValuationsView />} />
              <Route path="/adp" element={<ADPReportView />} />
              <Route path="/players" element={<PlayerSearchView />} />
              <Route path="/keeper-planner" element={<KeeperPlannerView />} />
            </Route>
          </Routes>
        </PlayerDrawerProvider>
      </DraftSessionProvider>
    </BrowserRouter>
  );
}
