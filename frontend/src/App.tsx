import { DraftBoardTable } from "./components/DraftBoardTable";

export default function App() {
  return (
    <div className="min-h-screen bg-white p-4">
      <h1 className="text-xl font-bold mb-4">Draft Board</h1>
      <DraftBoardTable season={2026} />
    </div>
  );
}
