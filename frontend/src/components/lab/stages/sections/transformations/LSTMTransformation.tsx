import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function LSTMTransformation({ stage, activation }: Props) {
  const gates = activation.gateValues;
  const timestep = Number(stage.params?.timestep ?? 1);
  if (!gates) {
    return <p className="text-xs" style={{ color: "var(--text-3)" }}>No gate values available for this timestep.</p>;
  }

  const avg = (arr: Float32Array) => Array.from(arr).reduce((a, b) => a + b, 0) / Math.max(1, arr.length);
  const forget = avg(gates.forget);
  const input = avg(gates.input);
  const output = avg(gates.output);

  return (
    <div className="space-y-3">
      <div className="rounded-lg border p-3" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
        <div className="mb-2 text-xs" style={{ color: "var(--text-3)" }}>Timestep t={timestep} memory update</div>
        <div className="grid gap-2 md:grid-cols-3">
          <Gate label="Forget gate" value={forget} color="var(--error)" />
          <Gate label="Input gate" value={input} color="var(--success)" />
          <Gate label="Output gate" value={output} color="var(--info)" />
        </div>
      </div>
      <div className="rounded-lg border p-2 text-[10px] font-mono" style={{ borderColor: "var(--glass-border)", background: "var(--bg-void)", color: "var(--text-2)" }}>
        h_t sample: [{Array.from(activation.outputData.slice(0, 10)).map((v) => v.toFixed(3)).join(", ")}, ...]
      </div>
    </div>
  );
}

function Gate({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="rounded border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-void)" }}>
      <div className="text-[10px] uppercase" style={{ color: "var(--text-4)" }}>{label}</div>
      <div className="my-1 h-2 overflow-hidden rounded" style={{ background: "var(--bg-panel)" }}>
        <div className="h-full" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }} />
      </div>
      <div className="text-xs font-mono" style={{ color }}>{(value * 100).toFixed(1)}%</div>
    </div>
  );
}
