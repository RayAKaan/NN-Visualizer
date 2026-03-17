import React, { useMemo, useState } from "react";
import { simulatorApi } from "../../hooks/useSimulatorApi";
import { useDatasetStore } from "../../store/datasetStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import type { FeatureMapResponse, FilterResponse, NeuronAtlasResponse, SaliencyResponse } from "../../types/simulator";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

function parseShape(text: string): number[] {
  return text
    .split(",")
    .map((v) => parseInt(v.trim()))
    .filter((v) => !Number.isNaN(v) && v > 0);
}

export function ActivationsView() {
  const { graphId, currentInput } = useSimulatorStore();
  const datasetId = useDatasetStore((s) => s.datasetId);
  const [shapeText, setShapeText] = useState("1,28,28");
  const [targetClass, setTargetClass] = useState(0);
  const [layerIndex, setLayerIndex] = useState(0);
  const [filterIndex, setFilterIndex] = useState(0);
  const [featureMaps, setFeatureMaps] = useState<FeatureMapResponse | null>(null);
  const [saliency, setSaliency] = useState<SaliencyResponse | null>(null);
  const [filterResponse, setFilterResponse] = useState<FilterResponse | null>(null);
  const [neuronAtlas, setNeuronAtlas] = useState<NeuronAtlasResponse | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const inputShape = useMemo(() => parseShape(shapeText), [shapeText]);
  const inputValues = useMemo(() => {
    const total = inputShape.reduce((acc, v) => acc * v, 1);
    if (currentInput && currentInput.length === total) return currentInput;
    return Array.from({ length: total }, () => 0);
  }, [currentInput, inputShape]);

  const handleFeatureMaps = async () => {
    if (!graphId) return;
    setStatus("Computing feature maps...");
    try {
      const res = await simulatorApi.featureMaps(graphId, inputValues, inputShape);
      setFeatureMaps(res);
      setStatus(null);
    } catch {
      setStatus("Failed to compute feature maps.");
    }
  };

  const handleSaliency = async (method: "gradient" | "grad_cam") => {
    if (!graphId) return;
    setStatus("Computing saliency...");
    try {
      const res = await simulatorApi.saliency(graphId, inputValues, inputShape, targetClass, method);
      setSaliency(res);
      setStatus(null);
    } catch {
      setStatus("Failed to compute saliency.");
    }
  };

  const handleFilterResponse = async () => {
    if (!graphId || !datasetId) return;
    setStatus("Computing filter response...");
    try {
      const res = await simulatorApi.filterResponse(graphId, datasetId, layerIndex, filterIndex, 50);
      setFilterResponse(res);
      setStatus(null);
    } catch {
      setStatus("Failed to compute filter response.");
    }
  };

  const handleNeuronAtlas = async () => {
    if (!graphId || !datasetId) return;
    setStatus("Computing neuron atlas...");
    try {
      const res = await simulatorApi.neuronAtlas(graphId, datasetId, layerIndex, 100);
      setNeuronAtlas(res);
      setStatus(null);
    } catch {
      setStatus("Failed to compute neuron atlas.");
    }
  };

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Activation Explorer</div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Input Shape</label>
            <NeuralInput value={shapeText} onChange={(e) => setShapeText(e.target.value)} placeholder="1,28,28" />
          </div>
          <div className="view-field">
            <label className="view-label">Target Class</label>
            <NeuralInput type="number" value={targetClass} onChange={(e) => setTargetClass(parseInt(e.target.value) || 0)} />
          </div>
          <NeuralButton variant="secondary" onClick={handleFeatureMaps} disabled={!graphId}>
            Feature Maps
          </NeuralButton>
          <NeuralButton variant="secondary" onClick={() => handleSaliency("gradient")} disabled={!graphId}>
            Saliency
          </NeuralButton>
          <NeuralButton variant="secondary" onClick={() => handleSaliency("grad_cam")} disabled={!graphId}>
            Grad-CAM
          </NeuralButton>
        </div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Layer Index</label>
            <NeuralInput type="number" value={layerIndex} onChange={(e) => setLayerIndex(parseInt(e.target.value) || 0)} />
          </div>
          <div className="view-field">
            <label className="view-label">Filter Index</label>
            <NeuralInput type="number" value={filterIndex} onChange={(e) => setFilterIndex(parseInt(e.target.value) || 0)} />
          </div>
          <NeuralButton variant="secondary" onClick={handleFilterResponse} disabled={!graphId || !datasetId}>
            Filter Response
          </NeuralButton>
          <NeuralButton variant="secondary" onClick={handleNeuronAtlas} disabled={!graphId || !datasetId}>
            Neuron Atlas
          </NeuralButton>
        </div>
        {status && <div className="view-status">{status}</div>}
      </NeuralPanel>

      {saliency && (
        <div className="view-grid">
          <NeuralPanel className="view-panel" variant="base">
            <div className="view-subtitle">Saliency Map</div>
            {saliency.saliency_base64 ? (
              <img src={saliency.saliency_base64} className="view-image" />
            ) : (
              <pre className="view-pre">{JSON.stringify(saliency.saliency_map[0]?.slice(0, 5))}</pre>
            )}
          </NeuralPanel>
          <NeuralPanel className="view-panel" variant="base">
            <div className="view-subtitle">Overlay</div>
            {saliency.overlay_base64 ? (
              <img src={saliency.overlay_base64} className="view-image" />
            ) : (
              <div className="view-text">No overlay available</div>
            )}
          </NeuralPanel>
        </div>
      )}

      {featureMaps && (
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Feature Maps</div>
          <div className="view-stack">
            {featureMaps.per_layer.map((layer) => (
              <div key={layer.layer_index} className="view-block">
                <div className="view-text">
                  Layer {layer.layer_index + 1} ({layer.layer_type})
                </div>
                <div className="view-grid">
                  {layer.feature_maps.slice(0, 8).map((fm) => (
                    <div key={fm.filter_index} className="view-image-card">
                      {fm.map_base64 ? <img src={fm.map_base64} className="view-image" /> : <div className="view-text">Max {fm.max_activation.toFixed(2)}</div>}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </NeuralPanel>
      )}

      {filterResponse && (
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Filter Response</div>
          <div className="view-grid">
            {filterResponse.top_activating_samples.map((s) => (
              <div key={s.sample_index} className="view-image-card">
                {s.activation_map_base64 ? <img src={s.activation_map_base64} className="view-image" /> : null}
              </div>
            ))}
          </div>
        </NeuralPanel>
      )}

      {neuronAtlas && (
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Neuron Atlas</div>
          <pre className="view-pre">{JSON.stringify(neuronAtlas.neurons.slice(0, 10), null, 2)}</pre>
        </NeuralPanel>
      )}
    </div>
  );
}
