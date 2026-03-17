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
      <p className="text-xs" style={{ color: "var(--text-3)" }}>Raw input values enter the network.</p>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-lg border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
          <div className="mb-2 text-[10px] uppercase" style={{ color: "var(--text-4)" }}>Before {JSON.stringify(stage.inputShape)}</div>
          <div className="flex h-12 items-end gap-px overflow-hidden rounded" style={{ background: "var(--bg-void)" }}>
            {sampleIn.map((v, i) => <div key={i} className="flex-1" style={{ height: `${(Math.abs(v) / max) * 100}%`, background: "var(--fwd)" }} />)}
          </div>
        </div>
        <div className="rounded-lg border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
          <div className="mb-2 text-[10px] uppercase" style={{ color: "var(--text-4)" }}>After {JSON.stringify(stage.outputShape)}</div>
          <div className="flex h-12 items-end gap-px overflow-hidden rounded" style={{ background: "var(--bg-void)" }}>
            {sampleOut.map((v, i) => <div key={i} className="flex-1" style={{ height: `${(Math.abs(v) / max) * 100}%`, background: "var(--activation-mid)" }} />)}
          </div>
        </div>
      </div>
    </div>
  );
}
