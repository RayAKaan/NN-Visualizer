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
    <div className="mt-3 rounded-xl border border-status-warning/25 bg-status-warning/[0.08] p-3">
      <div className="mb-2 text-xs text-status-warning">Stage Comparison: {stage.name}</div>
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <div className="text-[12px] text-ink-mute">Untrained</div>
          <div className="mt-1 flex h-12 gap-px overflow-hidden rounded">
            {u.map((v, i) => <div key={i} className="flex-1" style={{ background: `rgba(180,83,9,${Math.min(Math.abs(v), 1)})` }} />)}
          </div>
        </div>
        <div>
          <div className="text-[12px] text-ink-mute">Trained</div>
          <div className="mt-1 flex h-12 gap-px overflow-hidden rounded">
            {t.map((v, i) => <div key={i} className="flex-1" style={{ background: `rgba(0,114,178,${Math.min(Math.abs(v), 1)})` }} />)}
          </div>
        </div>
      </div>
      <div className="mt-2 text-xs text-ink-soft">Average activation difference: {diff.toFixed(4)}</div>
    </div>
  );
}
