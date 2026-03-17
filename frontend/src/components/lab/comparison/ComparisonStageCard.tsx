import type { StageActivation, StageDefinition } from "../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  trainedActivation: StageActivation;
  untrainedActivation: StageActivation | null;
}

export function ComparisonStageCard({ stage, trainedActivation, untrainedActivation }: Props) {
  if (!untrainedActivation) return null;
  const n = Math.min(64, trainedActivation.outputData.length, untrainedActivation.outputData.length);
  const t = Array.from(trainedActivation.outputData.slice(0, n));
  const u = Array.from(untrainedActivation.outputData.slice(0, n));
  let diff = 0;
  for (let i = 0; i < n; i += 1) diff += Math.abs(t[i] - u[i]);
  diff = n ? diff / n : 0;

  return (
    <div className="mt-3 rounded-xl border p-3" style={{ borderColor: "rgba(251,191,36,0.25)", background: "var(--status-warning-bg)" }}>
      <div className="mb-2 text-xs" style={{ color: "var(--status-warning)" }}>Stage Comparison: {stage.name}</div>
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <div className="text-[11px]" style={{ color: "var(--text-3)" }}>Untrained</div>
          <div className="mt-1 flex h-12 gap-px overflow-hidden rounded">
            {u.map((v, i) => <div key={i} className="flex-1" style={{ background: `rgba(251,191,36,${Math.min(Math.abs(v), 1)})` }} />)}
          </div>
        </div>
        <div>
          <div className="text-[11px]" style={{ color: "var(--text-3)" }}>Trained</div>
          <div className="mt-1 flex h-12 gap-px overflow-hidden rounded">
            {t.map((v, i) => <div key={i} className="flex-1" style={{ background: `rgba(56,189,248,${Math.min(Math.abs(v), 1)})` }} />)}
          </div>
        </div>
      </div>
      <div className="mt-2 text-xs" style={{ color: "var(--text-2)" }}>Average activation difference: {diff.toFixed(4)}</div>
    </div>
  );
}
