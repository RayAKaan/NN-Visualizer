import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function FlattenTransformation({ stage, activation }: Props) {
  return (
    <div className="space-y-3">
      <div className="rounded-lg border p-3" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
        <div className="text-xs" style={{ color: "var(--text-3)" }}>Flatten reshapes spatial data into a vector.</div>
        <div className="mt-2 grid gap-2 md:grid-cols-3">
          <Mini label="Input shape" value={JSON.stringify(activation.metadata.inputShape)} />
          <Mini label="Operation" value="reshape" />
          <Mini label="Output shape" value={JSON.stringify(activation.metadata.outputShape)} />
        </div>
      </div>
      <div className="rounded-lg border p-2 text-[10px] font-mono" style={{ borderColor: "var(--glass-border)", background: "var(--bg-void)", color: "var(--text-2)" }}>
        [{Array.from(activation.inputData.slice(0, 8)).map((v) => v.toFixed(3)).join(", ")}, ...] ? [{Array.from(activation.outputData.slice(0, 8)).map((v) => v.toFixed(3)).join(", ")}, ...]
      </div>
      <p className="text-[10px]" style={{ color: "var(--text-4)" }}>{JSON.stringify(stage.inputShape)} ? {JSON.stringify(stage.outputShape)}</p>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-void)" }}>
      <div className="text-[10px] uppercase" style={{ color: "var(--text-4)" }}>{label}</div>
      <div className="text-xs font-mono" style={{ color: "var(--text-1)" }}>{value}</div>
    </div>
  );
}
