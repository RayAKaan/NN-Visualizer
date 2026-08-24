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
    <div className="mt-5 rounded-2xl border border-ember-700/20 bg-ember-700/[0.06] p-4">
      <h3 className="text-sm font-semibold text-ink">Set Ground Truth Label</h3>
      <p className="mt-1 text-xs text-ink-mute">
        Predicted <b className="text-arch-ann">{String(finalPrediction.label)}</b>. Choose the true label to start backpropagation.
      </p>
      <div className={`mt-3 grid gap-2 ${dataset === "catdog" ? "grid-cols-2" : "grid-cols-5 sm:grid-cols-10"}`}>
        {classes.map((c) => (
          <button key={c.value} type="button" onClick={() => setTrueLabel(c.value)}
            className={`rounded-lg border px-2 py-2 text-xs ${trueLabel === c.value ? "border-ember-700 bg-ember-600/15 text-ember-800" : "border-barley-linestrong bg-white text-ink-soft hover:border-ember-600/40"}`}>
            {c.label}{predictedLabel === c.value ? " \u00b7 predicted" : ""}
          </button>
        ))}
      </div>
      <button type="button" disabled={trueLabel === null} onClick={() => void startBackwardPass()}
        className="mt-3 w-full rounded-xl bg-ember-700 px-3 py-2 text-sm font-semibold text-white shadow-ember transition-colors hover:bg-ember-800 disabled:opacity-40">
        Start Backward Pass
      </button>
    </div>
  );
}
