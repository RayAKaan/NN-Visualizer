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
      <div className="text-slate-400">Softmax probability distribution.</div>
      <div className="grid grid-cols-10 gap-1">
        {probs.map((p, idx) => {
          const label = dataset === "catdog" ? (idx === 0 ? "Cat" : "Dog") : String(idx);
          return (
            <div key={idx} className="rounded border border-slate-700 bg-slate-900/55 p-1 text-center">
              <div className="h-16 w-full rounded bg-slate-800/70">
                <div
                  className="mx-auto mt-auto h-full w-4 rounded-t"
                  style={{
                    transformOrigin: "bottom",
                    transform: `scaleY(${Math.max(0.01, p)})`,
                    background: idx === top ? "#22d3ee" : "rgba(148,163,184,0.7)",
                  }}
                />
              </div>
              <div className="mt-1 text-[10px] text-slate-300">{label}</div>
              <div className="text-[10px] font-mono text-slate-400">{(p * 100).toFixed(1)}%</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
