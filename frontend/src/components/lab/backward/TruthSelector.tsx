import { useLabStore } from "../../../store/labStore";

export function TruthSelector() {
  const finalPrediction = useLabStore((s) => s.finalPrediction);
  const dataset = useLabStore((s) => s.dataset);
  const trueLabel = useLabStore((s) => s.trueLabel);
  const setTrueLabel = useLabStore((s) => s.setTrueLabel);
  const startBackwardPass = useLabStore((s) => s.startBackwardPass);

  if (!finalPrediction) return null;

  const classes = dataset === "catdog"
    ? [{ value: 0, label: "Cat" }, { value: 1, label: "Dog" }]
    : Array.from({ length: 10 }, (_, i) => ({ value: i, label: String(i) }));

  const predictedLabel = dataset === "catdog"
    ? finalPrediction.label === "Cat" ? 0 : 1
    : Number(finalPrediction.label);

  return (
    <div className="mt-5 rounded-2xl border p-4" style={{ background: "var(--bwd-bg)", borderColor: "var(--bwd-border)" }}>
      <h3 className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>Set Ground Truth Label</h3>
      <p className="mt-1 text-xs" style={{ color: "var(--text-3)" }}>
        Predicted <b style={{ color: "var(--fwd)" }}>{String(finalPrediction.label)}</b>. Choose the true label to start backpropagation.
      </p>
      <div className={`mt-3 grid gap-2 ${dataset === "catdog" ? "grid-cols-2" : "grid-cols-5 sm:grid-cols-10"}`}>
        {classes.map((c) => (
          <button key={c.value} type="button" onClick={() => setTrueLabel(c.value)}
            className="rounded-lg border px-2 py-2 text-xs"
            style={{
              borderColor: trueLabel === c.value ? "var(--bwd)" : "var(--glass-border)",
              background: trueLabel === c.value ? "rgba(251,146,60,0.18)" : "var(--bg-card)",
              color: trueLabel === c.value ? "var(--bwd)" : "var(--text-2)",
            }}>
            {c.label}{predictedLabel === c.value ? " • predicted" : ""}
          </button>
        ))}
      </div>
      <button type="button" disabled={trueLabel === null} onClick={() => void startBackwardPass()}
        className="mt-3 w-full rounded-xl px-3 py-2 text-sm font-semibold disabled:opacity-40"
        style={{ background: "var(--bwd)", color: "var(--text-inverse)" }}>
        Start Backward Pass
      </button>
    </div>
  );
}
