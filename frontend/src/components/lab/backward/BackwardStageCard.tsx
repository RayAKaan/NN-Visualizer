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
    <div className="rounded-2xl border" style={{ borderColor: expanded ? "var(--bwd-border)" : "var(--glass-border)", background: expanded ? "var(--bwd-bg)" : "var(--bg-card)", opacity: status === "locked" ? 0.45 : 1 }}>
      <div className="flex items-center gap-3 px-4 py-3">
        <div className="grid h-8 w-8 place-items-center rounded-lg text-xs font-bold" style={{ background: "rgba(251,146,60,0.2)", color: "var(--bwd)" }}>
          ?{stageNumber}
        </div>
        <div className="flex-1">
          <div className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>Backward: {stage.name}</div>
          <div className="text-[11px]" style={{ color: "var(--text-4)" }}>{status}</div>
        </div>
      </div>

      {expanded && (
        <div className="space-y-3 px-4 pb-4">
          <div className="rounded-lg border p-3" style={{ borderColor: "var(--math-panel-border)", background: "var(--math-panel-bg)" }}>
            <div className="text-[11px]" style={{ color: "var(--text-3)" }}>{eq.explanation}</div>
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
