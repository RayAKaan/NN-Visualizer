import type { Dataset, StageActivation } from "../../../../types/pipeline";

interface Props {
  activation: StageActivation;
  dataset: Dataset;
}

export function PreprocessingViz({ activation, dataset }: Props) {
  const sample = Array.from(activation.outputData.slice(0, 32));
  return (
    <div className="space-y-2 text-xs">
      <div className="text-slate-400">{dataset === "mnist" ? "Normalized grayscale" : "Normalized RGB tensors"} preview.</div>
      <div className="grid grid-cols-8 gap-1">
        {sample.map((v, idx) => (
          <div key={idx} className="h-5 rounded" style={{ background: `rgba(56,189,248,${0.08 + Math.min(Math.abs(v), 1) * 0.92})` }} />
        ))}
      </div>
    </div>
  );
}
