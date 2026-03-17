import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function SoftmaxTransformation({ activation }: Props) {
  const probs = Array.from(activation.outputData);
  const top = probs.reduce((best, v, i, arr) => (v > arr[best] ? i : best), 0);
  const max = Math.max(...probs, 0.01);

  return (
    <div className="space-y-3">
      <div className="rounded-lg border p-3" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
        <div className="mb-2 text-xs" style={{ color: "var(--text-3)" }}>Softmax normalizes logits into probabilities.</div>
        <div className="space-y-1">
          {probs.map((p, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="w-8 text-right font-mono" style={{ color: i === top ? "var(--fwd)" : "var(--text-4)" }}>{i}</span>
              <div className="h-3 flex-1 overflow-hidden rounded" style={{ background: "var(--bg-void)" }}>
                <div className="h-full" style={{ width: `${(p / max) * 100}%`, background: i === top ? "var(--fwd)" : "var(--text-4)" }} />
              </div>
              <span className="w-16 text-right font-mono" style={{ color: i === top ? "var(--fwd)" : "var(--text-3)" }}>{(p * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
