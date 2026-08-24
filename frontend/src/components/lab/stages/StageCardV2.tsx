import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { MathRenderer } from "../MathRenderer";
import type { StageActivation, StageDefinition, StageStatus } from "../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  status: StageStatus;
  activation: StageActivation | null;
  isCurrent: boolean;
  stageNumber: number;
  totalStages: number;
}

export function StageCardV2({ stage, status, activation, isCurrent, stageNumber, totalStages }: Props) {
  const formatShape = (shape: number[]) => {
    if (shape.length === 0) return "Scalar";
    return `[${shape.join(" \u00d7 ")}]`;
  };

  const isExpanded = status === "active" || status === "processing" || status === "completed";
  const panelVariant = isCurrent ? "elevated" : "base";

  return (
    <NeuralPanel variant={panelVariant} className={`transition-all duration-300 w-full ${status === "locked" ? "opacity-50 grayscale-[50%]" : "opacity-100"}`}>
      <div className="flex w-full items-center gap-4 p-4">
        <div className={`grid h-10 w-10 shrink-0 place-items-center rounded-xl font-bold ${isCurrent ? "bg-ember-600/15 text-ember-700 shadow-[0_1px_6px_rgba(194,65,12,0.3)]" : "bg-barley-wash text-ink-mute"}`}>
          {stageNumber}
        </div>

        <div className="flex w-full flex-col justify-center min-w-0">
          <div className="flex items-center justify-between gap-4">
            <h3 className="truncate text-base font-semibold text-ink">{stage.name}</h3>
            <span className="shrink-0 rounded bg-barley-wash px-2 py-0.5 text-[12px] font-medium tracking-wider text-ink-mute uppercase">
              {stage.type}
            </span>
          </div>
          <div className="mt-1 flex items-center gap-3 text-xs text-ink-mute">
            <span className="font-mono">{formatShape(stage.inputShape)} &rarr; {formatShape(stage.outputShape)}</span>
            {status === "processing" && <span className="flex items-center gap-1 text-ember-700"><div className="h-1.5 w-1.5 animate-pulse rounded-full bg-ember-700" /> Computing...</span>}
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="border-t border-barley-linestrong bg-white p-4">
          <div className="mb-4 rounded-xl border border-barley-linestrong bg-barley-page p-3 shadow-inner">
            <p className="mb-2 text-xs leading-relaxed text-ink-soft">{stage.explanation}</p>
            <div className="overflow-x-auto pb-1">
               <MathRenderer latex={stage.equations.primary} displayMode />
            </div>
            {stage.equations.explanation && (
              <p className="mt-2 text-[12px] text-ink-faint">{stage.equations.explanation}</p>
            )}
          </div>

          {activation && (
            <div className="text-xs text-ink-mute font-mono flex items-center justify-between bg-barley-sunken p-2 rounded-lg border border-barley-linestrong">
               <span>Params: {activation.metadata.paramCount.toLocaleString()}</span>
               <span>Time: {activation.metadata.computeTimeMs.toFixed(2)}ms</span>
            </div>
          )}
        </div>
      )}
    </NeuralPanel>
  );
}
