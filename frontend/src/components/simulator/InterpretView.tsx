import React from "react";
import { useInterpretStore } from "../../store/interpretStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { useDatasetStore } from "../../store/datasetStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function InterpretView() {
  const graphId = useSimulatorStore((s) => s.graphId);
  const currentInput = useSimulatorStore((s) => s.currentInput);
  const trainData = useDatasetStore((s) => s.trainData);
  const { result, auxImage, loading, targetClass, nSteps, setTargetClass, setNSteps, compute } = useInterpretStore();

  const input = currentInput ?? (trainData[0]?.x ?? null);
  const disabled = !graphId || !input || loading;

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Interpretability</div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Target Class</label>
            <NeuralInput type="number" min={0} value={targetClass} onChange={(e) => setTargetClass(Number(e.target.value))} />
          </div>
          <div className="view-field">
            <label className="view-label">Steps</label>
            <NeuralInput type="number" min={10} max={200} value={nSteps} onChange={(e) => setNSteps(Number(e.target.value))} />
          </div>
          <NeuralButton
            variant="primary"
            disabled={disabled}
            onClick={() => {
              if (graphId && input) void compute(graphId, input, null);
            }}
          >
            {loading ? "Computing..." : "Compute"}
          </NeuralButton>
        </div>
        {!graphId && <div className="view-warning">Build a graph first.</div>}
        {!input && <div className="view-warning">Run a forward pass or load a dataset sample.</div>}
      </NeuralPanel>

      <div className="view-grid">
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Attribution Map</div>
          {result?.attributions_base64 || auxImage ? (
            <div className="view-image-wrap">
              <img src={result?.attributions_base64 ?? auxImage ?? ""} alt="Attribution heatmap" className="view-image" />
            </div>
          ) : (
            <div className="view-empty">No attribution map yet.</div>
          )}
        </NeuralPanel>
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Stats</div>
          {result ? (
            <div className="view-stack">
              <div className="view-text">Input output: {result.input_output.toFixed(4)}</div>
              <div className="view-text">Baseline output: {result.baseline_output.toFixed(4)}</div>
              <div className="view-text">Attribution sum: {result.attribution_sum.toFixed(4)}</div>
              <div className="view-text">Convergence delta: {result.convergence_delta.toFixed(4)}</div>
            </div>
          ) : (
            <div className="view-empty">No stats available.</div>
          )}
        </NeuralPanel>
      </div>
    </div>
  );
}
