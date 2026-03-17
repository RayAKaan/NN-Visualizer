import React from "react";
import { useAdversarialStore } from "../../store/adversarialStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { useDatasetStore } from "../../store/datasetStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralSelect } from "@/design-system/components/NeuralSelect";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

function inferSquareShape(length: number): number[] | null {
  const side = Math.sqrt(length);
  if (Number.isInteger(side)) return [1, side, side];
  return null;
}

export function AdversarialView() {
  const graphId = useSimulatorStore((s) => s.graphId);
  const datasetId = useDatasetStore((s) => s.datasetId);
  const trainData = useDatasetStore((s) => s.trainData);
  const { loading, method, epsilon, pgdSteps, pgdStepSize, result, setMethod, setEpsilon, setPgdSteps, setPgdStepSize, attack, evaluate, robustness } = useAdversarialStore();

  const sample = trainData[0];
  const input = sample?.x ?? null;
  const label = sample?.y ? (Array.isArray(sample.y) ? sample.y.findIndex((v) => v === Math.max(...sample.y)) : sample.y) : 0;
  const shape = input ? inferSquareShape(input.length) : null;

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Adversarial Lab</div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Method</label>
            <NeuralSelect value={method} onChange={(e) => setMethod(e.target.value as "fgsm" | "pgd")}>
              <option value="fgsm">FGSM</option>
              <option value="pgd">PGD</option>
            </NeuralSelect>
          </div>
          <div className="view-field">
            <label className="view-label">Epsilon</label>
            <NeuralInput type="number" step={0.01} value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} />
          </div>
          {method === "pgd" && (
            <>
              <div className="view-field">
                <label className="view-label">Steps</label>
                <NeuralInput type="number" value={pgdSteps} onChange={(e) => setPgdSteps(Number(e.target.value))} />
              </div>
              <div className="view-field">
                <label className="view-label">Step size</label>
                <NeuralInput type="number" step={0.01} value={pgdStepSize} onChange={(e) => setPgdStepSize(Number(e.target.value))} />
              </div>
            </>
          )}
          <NeuralButton
            variant="primary"
            disabled={!graphId || !input || loading}
            onClick={() => graphId && input && attack({ graph_id: graphId, input, true_label: Number(label), input_shape: shape })}
          >
            {loading ? "Running..." : "Run Attack"}
          </NeuralButton>
          <NeuralButton
            variant="secondary"
            disabled={!graphId || !datasetId || loading}
            onClick={() => graphId && datasetId && evaluate({ graph_id: graphId, dataset_id: datasetId, epsilons: [0.0, 0.05, 0.1, 0.15, 0.2] })}
          >
            Evaluate
          </NeuralButton>
        </div>
        {(!graphId || !input) && <div className="view-warning">Build a graph and load a dataset sample.</div>}
      </NeuralPanel>

      <div className="view-grid">
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Adversarial</div>
          {result?.adversarial_image_base64 ? (
            <img src={result.adversarial_image_base64} alt="Adversarial" className="view-image" />
          ) : (
            <div className="view-empty">No adversarial image yet.</div>
          )}
        </NeuralPanel>
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Perturbation</div>
          {result?.perturbation_amplified_base64 ? (
            <img src={result.perturbation_amplified_base64} alt="Perturbation" className="view-image" />
          ) : (
            <div className="view-empty">No perturbation yet.</div>
          )}
        </NeuralPanel>
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Stats</div>
          {result ? (
            <div className="view-stack">
              <div className="view-text">Original class: {result.original_prediction.class}</div>
              <div className="view-text">Adversarial class: {result.adversarial_prediction.class}</div>
              <div className="view-text">Attack success: {result.attack_success ? "Yes" : "No"}</div>
              <div className="view-text">Linf norm: {result.perturbation_stats.linf_norm.toFixed(4)}</div>
              <div className="view-text">L2 norm: {result.perturbation_stats.l2_norm.toFixed(4)}</div>
            </div>
          ) : (
            <div className="view-empty">No stats yet.</div>
          )}
        </NeuralPanel>
      </div>

      {robustness && (
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Robustness Curve</div>
          <div className="view-grid">
            {robustness.robustness_curve.map((r) => (
              <div key={r.epsilon} className="view-chip">
                epsilon {r.epsilon.toFixed(2)} - acc {Math.round(r.accuracy * 100)}%
              </div>
            ))}
          </div>
        </NeuralPanel>
      )}
    </div>
  );
}
