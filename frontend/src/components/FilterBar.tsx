const POSITIONS = ["C", "1B", "2B", "SS", "3B", "OF", "SP", "RP"] as const;
const PLAYER_TYPES = ["all", "batter", "pitcher"] as const;

export type PlayerTypeFilter = (typeof PLAYER_TYPES)[number];

interface FilterBarProps {
  playerTypeFilter: PlayerTypeFilter;
  onPlayerTypeChange: (value: PlayerTypeFilter) => void;
  positionFilter: string | null;
  onPositionChange: (value: string | null) => void;
}

export function FilterBar({
  playerTypeFilter,
  onPlayerTypeChange,
  positionFilter,
  onPositionChange,
}: FilterBarProps) {
  return (
    <div className="flex flex-wrap gap-2 items-center">
      <div className="flex gap-1">
        {PLAYER_TYPES.map((pt) => (
          <button
            key={pt}
            onClick={() => onPlayerTypeChange(pt)}
            className={`px-3 py-1 text-sm rounded ${
              playerTypeFilter === pt
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {pt === "all" ? "All" : pt === "batter" ? "Batters" : "Pitchers"}
          </button>
        ))}
      </div>

      <div className="w-px h-6 bg-gray-300" />

      <div className="flex gap-1">
        <button
          onClick={() => onPositionChange(null)}
          className={`px-2 py-1 text-xs rounded ${
            positionFilter === null
              ? "bg-blue-600 text-white"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          All
        </button>
        {POSITIONS.map((pos) => (
          <button
            key={pos}
            onClick={() => onPositionChange(pos)}
            className={`px-2 py-1 text-xs rounded ${
              positionFilter === pos
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {pos}
          </button>
        ))}
      </div>
    </div>
  );
}
