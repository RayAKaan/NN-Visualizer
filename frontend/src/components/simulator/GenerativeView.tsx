import React from "react";
import { useGenerativeStore } from "../../store/generativeStore";
import { useDatasetStore } from "../../store/datasetStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralSelect } from "@/design-system/components/NeuralSelect";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function GenerativeView() {
  const datasetId = useDatasetStore((s) => s.datasetId);
  const { loading, mode, nSamples, size, result, epochs, setMode, setNSamples, setSize, setEpochs, train, sample } = useGenerativeStore();

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Generative Playground</div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Mode</label>
            <NeuralSelect value={mode} onChange={(e) => setMode(e.target.value as "vae" | "gan" | "ae")}>
              <option value="vae">VAE</option>
              <option value="gan">GAN</option>
              <option value="ae">Autoencoder</option>
            </NeuralSelect>
          </div>
          <div className="view-field">
            <label className="view-label">Samples</label>
            <NeuralInput type="number" min={1} max={32} value={nSamples} onChange={(e) => setNSamples(Number(e.target.value))} />
          </div>
          <div className="view-field">
            <label className="view-label">Epochs</label>
            <NeuralInput type="number" min={1} max={50} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
          </div>
          <div className="view-field">
            <label className="view-label">Size</label>
            <NeuralInput type="number" min={8} max={64} value={size} onChange={(e) => setSize(Number(e.target.value))} />
          </div>
          <NeuralButton variant="secondary" disabled={!datasetId || loading} onClick={() => datasetId && train(datasetId)}>
            Train
          </NeuralButton>
          <NeuralButton variant="primary" disabled={loading} onClick={() => sample()}>
            {loading ? "Sampling..." : "Sample"}
          </NeuralButton>
        </div>
      </NeuralPanel>

      <NeuralPanel className="view-panel" variant="base">
        <div className="view-subtitle">Samples</div>
        <div className="view-grid">
          {(result?.samples ?? []).map((img, idx) => (
            <img key={idx} src={img} alt={`sample-${idx}`} className="view-image" />
          ))}
          {!result && <div className="view-empty">No samples yet.</div>}
        </div>
      </NeuralPanel>
    </div>
  );
}
