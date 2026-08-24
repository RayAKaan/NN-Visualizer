import type { Dataset, StageActivation, StageDefinition } from "../../../../types/pipeline";

interface Props {
  activation: StageActivation;
  stage: StageDefinition;
  dataset: Dataset;
}

export function SoftmaxViz({ activation, dataset }: Props) {
  const probs = Array.from(activation.outputData);
  const top = probs.reduce((best, value, idx, arr) => (value > arr[best] ? idx : best), 0);
  return (
    <div className="space-y-2 text-xs">
      <div className="text-ink-mute">Softmax probability distribution.</div>
      <div className="grid grid-cols-10 gap-1">
        {probs.map((p, idx) => {
          const label = dataset === "catdog" ? (idx === 0 ? "Cat" : "Dog") : String(idx);
          return (
            <div key={idx} className="rounded border border-barley-line bg-white p-1 text-center">
              <div className="h-16 w-full rounded bg-barley-wash">
                <div
                  className="mx-auto mt-auto h-full w-4 rounded-t"
                  style={{
                    transformOrigin: "bottom",
                    transform: `scaleY(${Math.max(0.01, p)})`,
                    background: idx === top ? "#EA580C" : "rgba(28,25,23,0.18)",
                  }}
                />
              </div>
              <div className="mt-1 text-[12px] text-ink-soft">{label}</div>
              <div className="text-[12px] font-mono text-ink-mute">{(p * 100).toFixed(1)}%</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
