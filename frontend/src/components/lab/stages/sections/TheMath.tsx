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
    <section className="overflow-hidden rounded-xl border" style={{ borderColor: "var(--math-panel-border)", background: "var(--math-panel-bg)" }}>
      <div className="border-b px-4 py-2 text-xs font-semibold uppercase" style={{ color: "var(--fwd)", borderColor: "var(--glass-border)" }}>The Math</div>
      <div className="space-y-3 p-4">
        <div className="rounded-lg p-2" style={{ background: "rgba(0,0,0,0.2)" }}>
          <MathRenderer latex={stage.equations.primary} displayMode />
        </div>
        <div
          className="rounded-lg border p-2"
          style={{ background: "rgba(0,0,0,0.2)", borderColor: highlightedVariable ? "var(--fwd-border)" : "var(--glass-border)" }}
          onMouseEnter={() => onHighlightVariable("math")}
          onMouseLeave={() => onHighlightVariable(null)}
        >
          <MathRenderer latex={live} displayMode />
        </div>
        <div className="flex gap-3 text-xs" style={{ color: "var(--text-3)" }}>
          <span className="font-mono">Params: {activation.metadata.paramCount.toLocaleString()}</span>
          <span className="font-mono">Time: {activation.metadata.computeTimeMs.toFixed(2)}ms</span>
        </div>
      </div>
    </section>
  );
}
