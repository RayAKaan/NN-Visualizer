import { MathRenderer } from "../MathRenderer";
import { getBackwardEquation } from "../../../data/equations/backward";
import type { BackwardStageActivation, StageDefinition, StageStatus } from "../../../types/pipeline";
import { GradientFlowViz } from "./GradientFlowViz";

interface Props {
  stage: StageDefinition;
  status: StageStatus;
  activation: BackwardStageActivation | null;
  stageNumber: number;
}

export function BackwardStageCard({ stage, status, activation, stageNumber }: Props) {
  const expanded = status === "active" || status === "processing";
  const eq = getBackwardEquation(stage);

  return (
    <div
      className={`rounded-2xl border ${expanded ? "border-ember-700/20 bg-ember-700/[0.06]" : "border-barley-linestrong bg-white"}`}
      style={{ opacity: status === "locked" ? 0.45 : 1 }}
    >
      <div className="flex items-center gap-3 px-4 py-3">
        <div className="grid h-8 w-8 place-items-center rounded-lg bg-ember-600/15 text-xs font-bold text-ember-800">
          &larr;{stageNumber}
        </div>
        <div className="flex-1">
          <div className="text-sm font-semibold text-ink">Backward: {stage.name}</div>
          <div className="text-[12px] text-ink-faint">{status}</div>
        </div>
      </div>

      {expanded && (
        <div className="space-y-3 px-4 pb-4">
          <div className="rounded-lg border border-barley-line bg-barley-sunken p-3">
            <div className="text-[12px] text-ink-mute">{eq.explanation}</div>
            <MathRenderer latex={eq.chainRule} displayMode />
            <MathRenderer latex={eq.localGradient} displayMode />
            {eq.weightGradient ? <MathRenderer latex={eq.weightGradient} displayMode /> : null}
          </div>
          {activation ? <GradientFlowViz activation={activation} stage={stage} /> : null}
        </div>
      )}
    </div>
  );
}
