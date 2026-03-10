import { BrowserRouter, Routes, Route } from "react-router-dom";
import { DraftDashboard } from "./components/DraftDashboard";
import { DraftSessionProvider } from "./context/DraftSessionContext";
import { PlayerDrawerProvider } from "./context/PlayerDrawerContext";
import { AppLayout } from "./components/AppLayout";
import { ProjectionsView } from "./components/ProjectionsView";
import { ValuationsView } from "./components/ValuationsView";
import { ADPReportView } from "./components/ADPReportView";
import { PlayerSearchView } from "./components/PlayerSearchView";

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
            </Route>
          </Routes>
        </PlayerDrawerProvider>
      </DraftSessionProvider>
    </BrowserRouter>
  );
}
