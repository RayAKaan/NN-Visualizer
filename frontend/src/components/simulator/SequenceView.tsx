import React, { useState } from "react";
import { simulatorApi } from "../../hooks/useSimulatorApi";
import { useSimulatorStore } from "../../store/simulatorStore";
import type { SequenceFullResponse, SequenceStepResponse } from "../../types/simulator";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

const defaultSequence = JSON.stringify(
  [
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.2, 0.1],
  ],
  null,
  2,
);

export function SequenceView() {
  const { graphId } = useSimulatorStore();
  const [sequenceText, setSequenceText] = useState(defaultSequence);
  const [timestep, setTimestep] = useState(0);
  const [stepResult, setStepResult] = useState<SequenceStepResponse | null>(null);
  const [fullResult, setFullResult] = useState<SequenceFullResponse | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const parseSequence = () => {
    try {
      const parsed = JSON.parse(sequenceText);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  };

  const handleStep = async () => {
    if (!graphId) return;
    const seq = parseSequence();
    setStatus("Running step...");
    try {
      const res = await simulatorApi.sequenceStep(graphId, seq, timestep);
      setStepResult(res);
      setStatus(null);
    } catch {
      setStatus("Failed to run step.");
    }
  };

  const handleFull = async () => {
    if (!graphId) return;
    const seq = parseSequence();
    setStatus("Running full sequence...");
    try {
      const res = await simulatorApi.sequenceFull(graphId, seq);
      setFullResult(res);
      setStatus(null);
    } catch {
      setStatus("Failed to run full sequence.");
    }
  };

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Sequence Simulator</div>
        <div className="view-section">
          <label className="view-label">Sequence (JSON)</label>
          <textarea
            className="view-textarea"
            value={sequenceText}
            onChange={(e) => setSequenceText(e.target.value)}
          />
        </div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Timestep</label>
            <NeuralInput
              type="number"
              value={timestep}
              onChange={(e) => setTimestep(parseInt(e.target.value) || 0)}
            />
          </div>
          <NeuralButton variant="secondary" onClick={handleStep} disabled={!graphId}>
            Step
          </NeuralButton>
          <NeuralButton variant="primary" onClick={handleFull} disabled={!graphId}>
            Run Full
          </NeuralButton>
        </div>
        {status && <div className="view-status">{status}</div>}
      </NeuralPanel>

      {stepResult && (
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Step {stepResult.timestep}</div>
          <div className="view-text">
            Hidden: {stepResult.new_hidden.slice(0, 6).map((v) => v.toFixed(2)).join(", ")}
          </div>
          {stepResult.gates && (
            <div className="view-text">Gates: {Object.keys(stepResult.gates).join(", ")}</div>
          )}
        </NeuralPanel>
      )}

      {fullResult && (
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Full Sequence Output</div>
          <div className="view-text">Hidden states: {fullResult.all_hidden_states.length}</div>
          {fullResult.attention_heatmap_base64 && (
            <img src={fullResult.attention_heatmap_base64} className="view-image" />
          )}
        </NeuralPanel>
      )}
    </div>
  );
}
