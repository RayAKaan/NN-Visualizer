import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function LSTMTransformation({ stage, activation }: Props) {
  const gates = activation.gateValues;
  const timestep = Number(stage.params?.timestep ?? 1);
  if (!gates) {
    return <p className="text-xs text-ink-mute">No gate values available for this timestep.</p>;
  }

  const avg = (arr: Float32Array) => Array.from(arr).reduce((a, b) => a + b, 0) / Math.max(1, arr.length);
  const forget = avg(gates.forget);
  const input = avg(gates.input);
  const output = avg(gates.output);

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-barley-linestrong bg-white p-3">
        <div className="mb-2 text-xs text-ink-mute">Timestep t={timestep} memory update</div>
        <div className="grid gap-2 md:grid-cols-3">
          <Gate label="Forget gate" value={forget} color="#B91C1C" />
          <Gate label="Input gate" value={input} color="#15803D" />
          <Gate label="Output gate" value={output} color="#0072B2" />
        </div>
      </div>
      <div className="rounded-lg border border-barley-linestrong bg-barley-page p-2 text-[12px] font-mono text-ink-soft">
        h_t sample: [{Array.from(activation.outputData.slice(0, 10)).map((v) => v.toFixed(3)).join(", ")}, ...]
      </div>
    </div>
  );
}

function Gate({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="rounded border border-barley-linestrong bg-barley-page p-2">
      <div className="text-[12px] uppercase text-ink-faint">{label}</div>
      <div className="my-1 h-2 overflow-hidden rounded bg-white">
        <div className="h-full" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }} />
      </div>
      <div className="text-xs font-mono" style={{ color }}>{(value * 100).toFixed(1)}%</div>
    </div>
  );
}
