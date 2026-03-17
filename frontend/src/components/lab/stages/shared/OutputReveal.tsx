import { useLabStore } from "../../../../store/labStore";

const COLORS: Record<string, string> = {
  ANN: "#f472b6",
  CNN: "#22d3ee",
  RNN: "#a855f7",
};

export function OutputReveal() {
  const pred = useLabStore((s) => s.finalPrediction);
  const architecture = useLabStore((s) => s.architecture);
  const accent = COLORS[architecture] ?? "#22d3ee";

  if (!pred) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-4 text-center text-xs text-slate-400">
        Prediction locked until all stages finish.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-emerald-400/35 bg-emerald-500/10 p-4 text-center">
      <div className="text-xs uppercase tracking-wide text-emerald-300/80">Journey complete</div>
      <div className="mt-1 text-6xl font-bold" style={{ color: accent, textShadow: `0 0 28px ${accent}55` }}>
        {typeof pred.label === "number" ? pred.label : pred.label === "Cat" ? "??" : "??"}
      </div>
      <div className="mt-1 text-sm text-slate-100">Predicted {String(pred.label)}</div>
      <div className="text-xs font-mono text-emerald-300">{pred.confidence.toFixed(2)}%</div>
    </div>
  );
}
