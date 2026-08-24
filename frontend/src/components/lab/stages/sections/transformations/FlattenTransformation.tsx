import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function FlattenTransformation({ stage, activation }: Props) {
  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-barley-linestrong bg-white p-3">
        <div className="text-xs text-ink-mute">Flatten reshapes spatial data into a vector.</div>
        <div className="mt-2 grid gap-2 md:grid-cols-3">
          <Mini label="Input shape" value={JSON.stringify(activation.metadata.inputShape)} />
          <Mini label="Operation" value="reshape" />
          <Mini label="Output shape" value={JSON.stringify(activation.metadata.outputShape)} />
        </div>
      </div>
      <div className="rounded-lg border border-barley-linestrong bg-barley-page p-2 text-[12px] font-mono text-ink-soft">
        [{Array.from(activation.inputData.slice(0, 8)).map((v) => v.toFixed(3)).join(", ")}, ...] &rarr; [{Array.from(activation.outputData.slice(0, 8)).map((v) => v.toFixed(3)).join(", ")}, ...]
      </div>
      <p className="text-[12px] text-ink-faint">
        {JSON.stringify(stage.inputShape)} &rarr; {JSON.stringify(stage.outputShape)}
      </p>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-barley-linestrong bg-barley-page p-2">
      <div className="text-[12px] uppercase text-ink-faint">{label}</div>
      <div className="text-xs font-mono text-ink">{value}</div>
    </div>
  );
}
