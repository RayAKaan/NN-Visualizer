import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function PreprocessTransformation({ stage, activation }: Props) {
  const before = activation.inputData.slice(0, 10);
  const after = activation.outputData.slice(0, 10);

  return (
    <div className="rounded-lg border p-3" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
      <div className="mb-2 text-xs" style={{ color: "var(--text-3)" }}>Normalization transforms raw values into a stable range.</div>
      <div className="overflow-auto text-xs font-mono" style={{ color: "var(--text-2)" }}>
        {Array.from(before).map((v, i) => (
          <div key={i} className="flex items-center gap-2">
            <span style={{ color: "var(--text-4)" }}>x[{i}]</span>
            <span>{v.toFixed(4)}</span>
            <span style={{ color: "var(--text-4)" }}>?</span>
            <span style={{ color: "var(--fwd)" }}>{after[i]?.toFixed(4)}</span>
          </div>
        ))}
      </div>
      <div className="mt-2 text-[10px]" style={{ color: "var(--text-4)" }}>
        {JSON.stringify(stage.inputShape)} ? {JSON.stringify(stage.outputShape)}
      </div>
    </div>
  );
}
