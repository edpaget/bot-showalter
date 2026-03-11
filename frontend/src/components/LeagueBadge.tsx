export function LeagueBadge({ leagueName, season }: { leagueName: string; season: number }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded bg-green-700 px-2 py-0.5 text-xs font-medium text-green-100">
      {leagueName}
      <span className="text-green-300">{season}</span>
    </span>
  );
}
