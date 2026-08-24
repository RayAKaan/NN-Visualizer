import { MathRenderer } from "../../MathRenderer";
import type { StageActivation, StageDefinition } from "../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
  level: "simple" | "technical" | "mathematical";
  highlightedVariable: string | null;
  onHighlightVariable: (v: string | null) => void;
}

function compact(values: Float32Array, n = 5): string {
  return Array.from(values.slice(0, n)).map((v) => v.toFixed(3)).join(", ");
}

export function TheMath({ stage, activation, highlightedVariable, onHighlightVariable }: Props) {
  const input = compact(activation.inputData);
  const output = compact(activation.outputData);
  const live = stage.type === "dense"
    ? `z = W x + b,\\; x=[${input},\\ldots],\\; z=[${output},\\ldots]`
    : stage.type === "activation_relu"
      ? `a = \\max(0,z),\\; z=[${input},\\ldots],\\; a=[${output},\\ldots]`
      : `input=[${input},\\ldots]\\to output=[${output},\\ldots]`;

  return (
    <section className="overflow-hidden rounded-xl border border-barley-line bg-barley-sunken">
      <div className="border-b border-barley-line px-4 py-2 text-xs font-semibold uppercase text-arch-ann">The Math</div>
      <div className="space-y-3 p-4">
        <div className="rounded-lg bg-white p-2 shadow-card">
          <MathRenderer latex={stage.equations.primary} displayMode />
        </div>
        <div
          className={`rounded-lg border bg-white p-2 ${highlightedVariable ? "border-arch-ann/40" : "border-barley-linestrong"}`}
          onMouseEnter={() => onHighlightVariable("math")}
          onMouseLeave={() => onHighlightVariable(null)}
        >
          <MathRenderer latex={live} displayMode />
        </div>
        <div className="flex gap-3 text-xs text-ink-mute">
          <span className="font-mono">Params: {activation.metadata.paramCount.toLocaleString()}</span>
          <span className="font-mono">Time: {activation.metadata.computeTimeMs.toFixed(2)}ms</span>
        </div>
      </div>
    </section>
  );
}
