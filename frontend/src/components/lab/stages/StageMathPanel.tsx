import { MathRenderer } from "../MathRenderer";
import type { EquationSet, StageActivation, StageDefinition } from "../../../types/pipeline";

interface Props {
  equations: EquationSet;
  activation: StageActivation | null;
  stage: StageDefinition;
}

function preview(arr: Float32Array): string {
  return Array.from(arr.slice(0, 4)).map((x) => x.toFixed(3)).join(", ");
}

function liveLatex(stage: StageDefinition, activation: StageActivation): string {
  const inp = preview(activation.inputData);
  const out = preview(activation.outputData);
  if (stage.type === "activation_relu") return `\\mathrm{ReLU}([${inp}, \\ldots])=[${out}, \\ldots]`;
  if (stage.type === "softmax") return `p=[${Array.from(activation.outputData).map((x) => x.toFixed(3)).join(",")}]`;
  if (stage.type === "dense") return `W\\cdot[${inp},\\ldots]+b=[${out},\\ldots]`;
  return `input=[${inp},\\ldots]\\to output=[${out},\\ldots]`;
}

export function StageMathPanel({ equations, activation, stage }: Props) {
  return (
    <div className="rounded-xl border border-violet-400/20 bg-violet-950/20 p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-violet-300/80">Mathematics</div>
      <div className="mt-2 rounded-md bg-black/35 p-3">
        <MathRenderer latex={equations.primary} displayMode />
      </div>
      {equations.explanation && <p className="mt-1 text-xs text-slate-400">{equations.explanation}</p>}
      {activation && (
        <>
          <div className="mt-2 rounded-md border border-emerald-400/20 bg-emerald-900/15 p-3">
            <MathRenderer latex={liveLatex(stage, activation)} displayMode />
          </div>
          <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-400">
            <span>Params: <span className="font-mono text-slate-300">{activation.metadata.paramCount}</span></span>
            <span>Compute: <span className="font-mono text-slate-300">{activation.metadata.computeTimeMs.toFixed(2)}ms</span></span>
          </div>
        </>
      )}
    </div>
  );
}
