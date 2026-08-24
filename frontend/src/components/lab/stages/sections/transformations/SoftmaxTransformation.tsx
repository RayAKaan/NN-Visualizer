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
      <div className="rounded-lg border border-barley-linestrong bg-white p-3">
        <div className="mb-2 text-xs text-ink-mute">Softmax normalizes logits into probabilities.</div>
        <div className="space-y-1">
          {probs.map((p, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className={`w-8 text-right font-mono ${i === top ? "text-arch-ann" : "text-ink-faint"}`}>{i}</span>
              <div className="h-3 flex-1 overflow-hidden rounded bg-barley-page">
                <div className="h-full" style={{ width: `${(p / max) * 100}%`, background: i === top ? "#0072B2" : "#D8CFC0" }} />
              </div>
              <span className={`w-16 text-right font-mono ${i === top ? "text-arch-ann" : "text-ink-mute"}`}>{(p * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
