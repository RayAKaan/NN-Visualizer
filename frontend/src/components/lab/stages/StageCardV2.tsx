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
        <div className={`grid h-10 w-10 shrink-0 place-items-center rounded-xl font-bold ${isCurrent ? "bg-cyan-500/20 text-cyan-300 shadow-[0_0_15px_rgba(34,211,238,0.25)]" : "bg-slate-800 text-slate-400"}`}>
          {stageNumber}
        </div>
        
        <div className="flex w-full flex-col justify-center min-w-0">
          <div className="flex items-center justify-between gap-4">
            <h3 className="truncate text-base font-semibold text-slate-100">{stage.name}</h3>
            <span className="shrink-0 rounded bg-slate-800/80 px-2 py-0.5 text-[10px] font-medium tracking-wider text-slate-400 uppercase">
              {stage.type}
            </span>
          </div>
          <div className="mt-1 flex items-center gap-3 text-xs text-slate-400">
            <span className="font-mono">{formatShape(stage.inputShape)} &rarr; {formatShape(stage.outputShape)}</span>
            {status === "processing" && <span className="flex items-center gap-1 text-cyan-400"><div className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400" /> Computing...</span>}
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="border-t border-slate-700/50 bg-slate-900/30 p-4">
          <div className="mb-4 rounded-xl border border-slate-700/50 bg-slate-950/50 p-3 shadow-inner">
            <p className="mb-2 text-xs leading-relaxed text-slate-300">{stage.explanation}</p>
            <div className="overflow-x-auto pb-1">
               <MathRenderer latex={stage.equations.primary} displayMode />
            </div>
            {stage.equations.explanation && (
              <p className="mt-2 text-[11px] text-slate-500">{stage.equations.explanation}</p>
            )}
          </div>
          
          {activation && (
            <div className="text-xs text-slate-400 font-mono flex items-center justify-between bg-black/40 p-2 rounded-lg border border-slate-700/50">
               <span>Params: {activation.metadata.paramCount.toLocaleString()}</span>
               <span>Time: {activation.metadata.computeTimeMs.toFixed(2)}ms</span>
            </div>
          )}
        </div>
      )}
    </NeuralPanel>
  );
}
