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
        <div className="rounded-lg border border-barley-linestrong bg-white p-2">
          <div className="mb-1 text-[12px] uppercase text-ink-faint">Before (z)</div>
          <div className="flex h-14 items-end gap-px overflow-hidden rounded bg-barley-page">
            {before.map((v, i) => (
              <div key={i} className="flex-1" style={{ height: `${(Math.abs(v) / max) * 100}%`, background: v >= 0 ? "#EA580C" : "rgba(0,114,178,0.55)" }} />
            ))}
          </div>
        </div>
        <div className="rounded-lg border border-barley-linestrong bg-white p-2">
          <div className="mb-1 text-[12px] uppercase text-ink-faint">After (a=max(0,z))</div>
          <div className="flex h-14 items-end gap-px overflow-hidden rounded bg-barley-page">
            {after.map((v, i) => (
              <div key={i} className="flex-1" style={{ height: `${Math.max((Math.abs(v) / max) * 100, 4)}%`, background: v > 0 ? "#EA580C" : "rgba(185,28,28,0.18)" }} />
            ))}
          </div>
        </div>
      </div>
      <div className="rounded-lg border border-barley-linestrong bg-white p-2 text-xs text-ink-soft">
        ReLU zeroed <strong className={pct > 65 ? "text-status-warning" : "text-status-success"}>{dead}/{n}</strong> values ({pct.toFixed(1)}%).
      </div>
    </div>
  );
}
