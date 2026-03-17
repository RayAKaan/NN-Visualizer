interface Props {
  bins: number[];
  counts: number[];
}

export function WeightDistribution({ bins, counts }: Props) {
  const max = Math.max(...counts, 1);
  return (
    <div>
      <div className="mb-2 text-xs" style={{ color: "var(--text-3)" }}>Weight Distribution</div>
      <div className="flex h-24 items-end gap-px rounded bg-slate-900/40 p-1">
        {counts.map((c, i) => (
          <div key={i} className="flex-1 rounded-t" style={{ height: `${(c / max) * 100}%`, background: "rgba(56,189,248,0.7)" }} title={`${bins[i]?.toFixed(3)} to ${bins[i + 1]?.toFixed(3)}: ${c}`} />
        ))}
      </div>
    </div>
  );
}
