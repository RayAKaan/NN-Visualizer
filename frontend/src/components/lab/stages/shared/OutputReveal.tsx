import { useLabStore } from "../../../../store/labStore";

const COLORS: Record<string, string> = {
  ANN: "#0072B2",
  CNN: "#00806A",
  RNN: "#A64D85",
};

export function OutputReveal() {
  const pred = useLabStore((s) => s.finalPrediction);
  const architecture = useLabStore((s) => s.architecture);
  const accent = COLORS[architecture] ?? "#0072B2";

  if (!pred) {
    return (
      <div className="rounded-xl border border-barley-linestrong bg-white p-4 text-center text-xs text-ink-mute">
        Prediction locked until all stages finish.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-status-success/40 bg-status-success/12 p-4 text-center">
      <div className="text-xs uppercase tracking-wide text-status-success">Journey complete</div>
      <div className="mt-1 text-6xl font-bold" style={{ color: accent, textShadow: `0 0 28px ${accent}55` }}>
        {typeof pred.label === "number" ? pred.label : pred.label === "Cat" ? "??" : "??"}
      </div>
      <div className="mt-1 text-sm text-ink">Predicted {String(pred.label)}</div>
      <div className="text-xs font-mono text-status-success">{pred.confidence.toFixed(2)}%</div>
    </div>
  );
}
