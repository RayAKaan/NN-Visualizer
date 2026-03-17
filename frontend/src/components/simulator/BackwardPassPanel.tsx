import React from "react";
import { useBackpropStore } from "../../store/backpropStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { NeuralBadge } from "@/design-system/components/NeuralBadge";

export function BackwardPassPanel() {
  const steps = useBackpropStore((s) => s.backwardSteps);
  const lossValue = useBackpropStore((s) => s.lossValue);
  const lossEquation = useBackpropStore((s) => s.lossEquation);
  const runBackward = useBackpropStore((s) => s.runBackward);
  const resetBackward = useBackpropStore((s) => s.resetBackward);
  const currentInput = useSimulatorStore((s) => s.currentInput);
  const currentTarget = useSimulatorStore((s) => s.currentTarget);

  const run = async () => {
    if (!currentInput || !currentTarget) return;
    await runBackward(currentInput, currentTarget);
  };

  return (
    <NeuralPanel className="pass-panel" variant="base">
      <div className="pass-header">
        <div className="pass-title">Backward Pass</div>
        <div className="pass-actions">
          <NeuralButton variant="primary" onClick={() => void run()}>
            Run Backward
          </NeuralButton>
          <NeuralButton variant="ghost" onClick={resetBackward}>
            Reset
          </NeuralButton>
        </div>
      </div>
      {lossValue != null && (
        <div className="pass-input">
          <NeuralBadge tone="danger">Loss</NeuralBadge>
          <span>
            {lossValue.toFixed(3)} - {lossEquation}
          </span>
        </div>
      )}
      <div className="pass-steps neural-scroll-area">
        {steps.map((s) => (
          <div key={s.step_index} className="pass-step">
            <div className="pass-step-title">{s.description}</div>
            <div className="pass-step-eq">{s.equation_text}</div>
          </div>
        ))}
        {steps.length === 0 && <div className="pass-empty">Run a backward pass to see steps.</div>}
      </div>
    </NeuralPanel>
  );
}
