import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function InputTransformation({ stage, activation }: Props) {
  const sampleIn = Array.from(activation.inputData.slice(0, 24));
  const sampleOut = Array.from(activation.outputData.slice(0, 24));
  const max = Math.max(...sampleIn.map((v) => Math.abs(v)), ...sampleOut.map((v) => Math.abs(v)), 0.01);

  return (
    <div className="space-y-3">
      <p className="text-xs text-ink-mute">Raw input values enter the network.</p>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-lg border border-barley-linestrong bg-white p-2">
          <div className="mb-2 text-[12px] uppercase text-ink-faint">Before {JSON.stringify(stage.inputShape)}</div>
          <div className="flex h-12 items-end gap-px overflow-hidden rounded bg-barley-page">
            {sampleIn.map((v, i) => (
              <div key={i} className="flex-1" style={{ height: `${(Math.abs(v) / max) * 100}%`, background: "#EA580C" }} />
            ))}
          </div>
        </div>
        <div className="rounded-lg border border-barley-linestrong bg-white p-2">
          <div className="mb-2 text-[12px] uppercase text-ink-faint">After {JSON.stringify(stage.outputShape)}</div>
          <div className="flex h-12 items-end gap-px overflow-hidden rounded bg-barley-page">
            {sampleOut.map((v, i) => (
              <div key={i} className="flex-1" style={{ height: `${Math.max((Math.abs(v) / max) * 100, 4)}%`, background: "#EA580C" }} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
