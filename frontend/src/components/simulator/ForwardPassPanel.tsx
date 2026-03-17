import React from "react";
import { useComputationStore } from "../../store/computationStore";
import { useDatasetStore } from "../../store/datasetStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { NeuralBadge } from "@/design-system/components/NeuralBadge";

export function ForwardPassPanel() {
  const steps = useComputationStore((s) => s.steps);
  const runFullForward = useComputationStore((s) => s.runFullForward);
  const reset = useComputationStore((s) => s.reset);
  const trainData = useDatasetStore((s) => s.trainData);
  const currentInput = useSimulatorStore((s) => s.currentInput);
  const setCurrentTarget = useSimulatorStore((s) => s.setCurrentTarget);
  const setCurrentInput = useSimulatorStore((s) => s.setCurrentInput);

  const runSample = async () => {
    if (trainData.length === 0) return;
    const pick = trainData[Math.floor(Math.random() * trainData.length)];
    setCurrentInput(pick.x);
    setCurrentTarget(pick.y);
    await runFullForward(pick.x);
  };

  return (
    <NeuralPanel className="pass-panel" variant="base">
      <div className="pass-header">
        <div className="pass-title">Forward Pass</div>
        <div className="pass-actions">
          <NeuralButton variant="primary" onClick={() => void runSample()}>
            New Sample
          </NeuralButton>
          <NeuralButton variant="ghost" onClick={reset}>
            Reset
          </NeuralButton>
        </div>
      </div>
      <div className="pass-input">
        <NeuralBadge tone="info">Input</NeuralBadge>
        <span>{currentInput ? `[${currentInput.map((v) => v.toFixed(2)).join(", ")}]` : "n/a"}</span>
      </div>
      <div className="pass-steps neural-scroll-area">
        {steps.map((s) => (
          <div key={s.step_index} className="pass-step">
            <div className="pass-step-title">{s.description}</div>
            <div className="pass-step-eq">{s.equation_text}</div>
            <div className="pass-step-out">Out: {s.output_values.slice(0, 6).map((v) => v.toFixed(3)).join(", ")}</div>
          </div>
        ))}
        {steps.length === 0 && <div className="pass-empty">Run a forward pass to see steps.</div>}
      </div>
    </NeuralPanel>
  );
}
