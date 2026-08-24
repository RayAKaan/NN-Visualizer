import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function PreprocessTransformation({ stage, activation }: Props) {
  const before = activation.inputData.slice(0, 10);
  const after = activation.outputData.slice(0, 10);

  return (
    <div className="rounded-lg border border-barley-linestrong bg-white p-3">
      <div className="mb-2 text-xs text-ink-mute">Normalization transforms raw values into a stable range.</div>
      <div className="overflow-auto text-xs font-mono text-ink-soft">
        {Array.from(before).map((v, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-ink-faint">x[{i}]</span>
            <span>{v.toFixed(4)}</span>
            <span className="text-ink-faint">&rarr;</span>
            <span className="text-arch-ann">{after[i]?.toFixed(4)}</span>
          </div>
        ))}
      </div>
      <div className="mt-2 text-[12px] text-ink-faint">
        {JSON.stringify(stage.inputShape)} &rarr; {JSON.stringify(stage.outputShape)}
      </div>
    </div>
  );
}
