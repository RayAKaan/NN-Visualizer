import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function ReLUTransformation({ activation }: Props) {
  const n = Math.min(64, activation.inputData.length, activation.outputData.length);
  const before = Array.from(activation.inputData.slice(0, n));
  const after = Array.from(activation.outputData.slice(0, n));
  const max = Math.max(...before.map((v) => Math.abs(v)), ...after.map((v) => Math.abs(v)), 0.01);
  const dead = after.filter((v) => v === 0).length;
  const pct = (dead / Math.max(1, n)) * 100;

  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-lg border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
          <div className="mb-1 text-[10px] uppercase" style={{ color: "var(--text-4)" }}>Before (z)</div>
          <div className="flex h-14 items-end gap-px overflow-hidden rounded" style={{ background: "var(--bg-void)" }}>
            {before.map((v, i) => <div key={i} className="flex-1" style={{ height: `${(Math.abs(v) / max) * 100}%`, background: v >= 0 ? "var(--fwd)" : "var(--weight-negative-mid)" }} />)}
          </div>
        </div>
        <div className="rounded-lg border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
          <div className="mb-1 text-[10px] uppercase" style={{ color: "var(--text-4)" }}>After (a=max(0,z))</div>
          <div className="flex h-14 items-end gap-px overflow-hidden rounded" style={{ background: "var(--bg-void)" }}>
            {after.map((v, i) => <div key={i} className="flex-1" style={{ height: `${(Math.abs(v) / max) * 100}%`, background: v > 0 ? "var(--fwd)" : "var(--status-error-bg)" }} />)}
          </div>
        </div>
      </div>
      <div className="rounded-lg border p-2 text-xs" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)", color: "var(--text-2)" }}>
        ReLU zeroed <strong style={{ color: pct > 65 ? "var(--warning)" : "var(--success)" }}>{dead}/{n}</strong> values ({pct.toFixed(1)}%).
      </div>
    </div>
  );
}
