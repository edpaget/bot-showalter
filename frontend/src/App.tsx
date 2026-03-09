import { DraftDashboard } from "./components/DraftDashboard";
import { DraftSessionProvider } from "./context/DraftSessionContext";

export default function App() {
  return (
    <DraftSessionProvider>
      <DraftDashboard season={2026} />
    </DraftSessionProvider>
  );
}
