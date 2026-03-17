import React, { useState } from "react";
import { simulatorApi } from "../../hooks/useSimulatorApi";
import { useArchitectureStore } from "../../store/architectureStore";
import { useDatasetStore } from "../../store/datasetStore";
import type { CompareSetupResponse, CompareResultsResponse } from "../../types/simulator";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function ComparisonView() {
  const { layers } = useArchitectureStore();
  const { datasetId } = useDatasetStore();
  const [epochs, setEpochs] = useState(20);
  const [status, setStatus] = useState<string | null>(null);
  const [setup, setSetup] = useState<CompareSetupResponse | null>(null);
  const [results, setResults] = useState<CompareResultsResponse | null>(null);

  const handleSetup = async () => {
    setStatus("Setting up comparison...");
    try {
      const res = await simulatorApi.compareSetup(
        [
          { model_id: "A", architecture: layers, config: { optimizer: "adam", learning_rate: 0.01 } },
          { model_id: "B", architecture: layers, config: { optimizer: "sgd", learning_rate: 0.01 } },
        ],
        datasetId ?? "",
        epochs,
      );
      setSetup(res);
      setStatus(null);
    } catch {
      setStatus("Failed to setup comparison.");
    }
  };

  const handleResults = async () => {
    if (!setup) return;
    setStatus("Fetching results...");
    try {
      const res = await simulatorApi.compareResults(setup.comparison_id);
      setResults(res);
      setStatus(null);
    } catch {
      setStatus("Failed to fetch results.");
    }
  };

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Model Comparison</div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Epochs</label>
            <NeuralInput type="number" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value) || 1)} />
          </div>
          <NeuralButton variant="secondary" onClick={handleSetup}>
            Setup
          </NeuralButton>
          <NeuralButton variant="primary" onClick={handleResults} disabled={!setup}>
            Results
          </NeuralButton>
        </div>
        {status && <div className="view-status">{status}</div>}
      </NeuralPanel>

      {setup && (
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Comparison ID</div>
          <div className="view-text">{setup.comparison_id}</div>
        </NeuralPanel>
      )}

      {results && (
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Results</div>
          <pre className="view-pre">{JSON.stringify(results, null, 2)}</pre>
        </NeuralPanel>
      )}
    </div>
  );
}
