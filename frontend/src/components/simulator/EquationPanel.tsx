import React from "react";
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import { useComputationStore } from "../../store/computationStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { neuralPalette } from "@/design-system/tokens/colors";

export function EquationPanel() {
  const equations = useComputationStore((s) => s.equations);
  const currentStepIndex = useSimulatorStore((s) => s.currentStepIndex);

  const toLatex = (value: string) =>
    value
      .replace(/\*/g, " \\cdot ")
      .replace(/·/g, " \\cdot ")
      .replace(/\s+/g, " ")
      .trim();

  const colorizeNumbers = (value: string) => {
    const numRegex = /-?\d+(\.\d+)?/g;
    return value.replace(numRegex, (match) => {
      const num = Number(match);
      if (Number.isNaN(num)) return match;
      const abs = Math.abs(num);
      const color = abs < 0.0001
        ? neuralPalette.ash
        : num >= 0
          ? neuralPalette.axon.bright
          : neuralPalette.dendrite.bright;
      return `\\color{${color}}{${match}}`;
    });
  };

  return (
    <div className="equations-panel">
      <div className="equations-title">Equations</div>
      {equations ? (
        <>
          <div className="equation-block equation-forward active">
            <div className="equation-label">Pre-Activation</div>
            <BlockMath math={toLatex(equations.generic_equations.pre_activation)} />
          </div>
          <div className="equation-block equation-forward">
            <div className="equation-label">Activation</div>
            <BlockMath math={toLatex(equations.generic_equations.activation)} />
          </div>
          <div className="equation-meta">
            W: {equations.dimensions.W_shape.join("x")} • Params: {equations.dimensions.param_count}
          </div>
          <div className="equation-meta">
            Mean {equations.weight_stats.mean.toFixed(3)} • Std {equations.weight_stats.std.toFixed(3)}
          </div>
          <div className="equation-block equation-forward numeric">
            <div className="equation-label">Numeric</div>
            {equations.numeric_equations.slice(0, 2).map((eq, idx) => (
              <div key={eq.neuron_index} className={idx === currentStepIndex ? "equation-active" : ""}>
                <BlockMath math={colorizeNumbers(toLatex(eq.pre_activation_eq))} />
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="equation-empty">Select a layer to view equations.</div>
      )}
    </div>
  );
}
