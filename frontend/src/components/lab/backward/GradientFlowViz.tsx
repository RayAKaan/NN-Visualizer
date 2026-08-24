import type { BackwardStageActivation, StageDefinition } from "../../../types/pipeline";

interface Props {
  activation: BackwardStageActivation;
  stage: StageDefinition;
}

export function GradientFlowViz({ activation, stage }: Props) {
  const outputGrads = Array.from(activation.outputGradient);
  const inputGrads = Array.from(activation.inputGradient);
  const max = Math.max(...outputGrads.map(Math.abs), ...inputGrads.map(Math.abs)) || 1;

  return (
    <div className="rounded-xl border border-barley-linestrong bg-barley-sunken p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs font-semibold text-ink-soft">Gradient flow: {stage.name}</div>
        <div className="flex items-center gap-1 text-[12px] text-ink-faint">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#C2410C" }} />
          |&part;L/&part;x|
        </div>
      </div>

      <div className="space-y-1">
        <div className="text-[12px] uppercase tracking-wider text-ink-faint">Output gradient</div>
        <div className="flex gap-px">
          {outputGrads.map((v, i) => (
            <div key={`o-${i}`} className="flex-1" style={{ background: `rgba(194,65,12,${Math.abs(v) / max})` }} />
          ))}
        </div>
      </div>

      <div className="mt-3 space-y-1">
        <div className="text-[12px] uppercase tracking-wider text-ink-faint">Input gradient</div>
        <div className="flex gap-px">
          {inputGrads.map((v, i) => (
            <div key={`i-${i}`} className="flex-1" style={{ background: `rgba(194,65,12,${Math.abs(v) / max})` }} />
          ))}
        </div>
      </div>
    </div>
  );
}
