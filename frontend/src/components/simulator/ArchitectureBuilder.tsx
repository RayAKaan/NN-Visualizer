import React from "react";
import { useArchitectureStore } from "../../store/architectureStore";
import { LayerCard } from "./LayerCard";
import { PresetSelector } from "./PresetSelector";
import type { LayerType } from "../../types/simulator";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";

const palette: Array<{ label: string; type: LayerType }> = [
  { label: "Dense", type: "dense" },
  { label: "Conv2D", type: "conv2d" },
  { label: "MaxPool", type: "maxpool2d" },
  { label: "AvgPool", type: "avgpool2d" },
  { label: "Flatten", type: "flatten" },
  { label: "BatchNorm", type: "batchnorm" },
  { label: "RNN", type: "rnn" },
  { label: "LSTM", type: "lstm" },
  { label: "GRU", type: "gru" },
  { label: "Embedding", type: "embedding" },
  { label: "Attention", type: "attention" },
  { label: "Residual", type: "residual" },
];

export function ArchitectureBuilder() {
  const layers = useArchitectureStore((s) => s.layers);
  const validation = useArchitectureStore((s) => s.validationResult);
  const totalParams = useArchitectureStore((s) => s.totalParams);
  const flops = useArchitectureStore((s) => s.flopsPerSample);
  const addLayerOfType = useArchitectureStore((s) => s.addLayerOfType);
  const removeLayer = useArchitectureStore((s) => s.removeLayer);
  const updateLayer = useArchitectureStore((s) => s.updateLayer);
  const build = useArchitectureStore((s) => s.build);

  return (
    <NeuralPanel className="architecture-panel" variant="base">
      <div className="arch-header">
        <div className="arch-title">Architecture Builder</div>
        <NeuralButton variant="primary" onClick={build} className="arch-build">
          Build
        </NeuralButton>
      </div>

      <div>
        <div className="arch-section-label">Layer Palette</div>
        <div className="arch-palette">
          {palette.map((item) => (
            <button
              key={item.type}
              onClick={() => addLayerOfType(item.type)}
              className="arch-add"
            >
              + {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="arch-layer-list">
        {layers.map((layer, i) => (
          <LayerCard
            key={`${layer.type}-${i}`}
            layer={layer}
            index={i}
            isRemovable={layer.type !== "input" && layer.type !== "output"}
            onRemove={() => removeLayer(i)}
            onChange={(updates) => updateLayer(i, updates)}
          />
        ))}
      </div>

      <PresetSelector />

      <div className="arch-meta">
        <div>Params: {totalParams}</div>
        <div>FLOPs/sample: {flops}</div>
        {validation?.errors?.length ? (
          <div className="arch-errors">Errors: {validation.errors.join("; ")}</div>
        ) : null}
      </div>
    </NeuralPanel>
  );
}
