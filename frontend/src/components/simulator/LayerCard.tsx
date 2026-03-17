import React, { useMemo } from "react";
import type { LayerConfig } from "../../types/simulator";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralSelect } from "@/design-system/components/NeuralSelect";
import { useSimulatorStore } from "../../store/simulatorStore";

interface Props {
  layer: LayerConfig;
  index: number;
  isRemovable: boolean;
  onRemove: () => void;
  onChange: (updates: Partial<LayerConfig>) => void;
}

const activationOptions = ["relu", "sigmoid", "tanh", "leaky_relu", "softmax", "linear"];
const initOptions = ["xavier", "he", "lecun", "random", "zeros"];

function parseShape(text: string): number[] {
  return text
    .split(",")
    .map((v) => parseInt(v.trim()))
    .filter((v) => !Number.isNaN(v) && v > 0);
}

export function LayerCard({ layer, index, isRemovable, onRemove, onChange }: Props) {
  const isInput = layer.type === "input";
  const isOutput = layer.type === "output";
  const typeLabel = isInput ? "Input" : isOutput ? "Output" : layer.type;
  const shapeText = useMemo(() => (layer.input_shape ? layer.input_shape.join(",") : ""), [layer.input_shape]);
  const selectedLayer = useSimulatorStore((s) => s.selectedLayerIndex);
  const setSelectedLayer = useSimulatorStore((s) => s.setSelectedLayerIndex);
  const isSelected = selectedLayer === index;

  return (
    <div
      className={`layer-card ${isInput ? "layer-card-input" : isOutput ? "layer-card-output" : "layer-card-hidden"} ${isSelected ? "selected" : ""}`}
      onClick={() => setSelectedLayer(index)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter") setSelectedLayer(index);
      }}
    >
      <div className="layer-card-header">
        <div className="layer-card-title">
          {typeLabel} {isInput || isOutput ? "" : `(Hidden ${index})`}
        </div>
        {isRemovable && (
          <button onClick={onRemove} className="layer-card-remove">
            ?
          </button>
        )}
      </div>

      {isInput ? (
        <div className="layer-card-row">
          <label className="layer-card-label">Input Shape</label>
          <NeuralInput
            value={shapeText}
            onChange={(e) => {
              const next = parseShape(e.target.value);
              onChange({ input_shape: next, neurons: next[0] ?? 1 });
            }}
            className="layer-card-input-control"
            placeholder="2 or 1,28,28"
          />
        </div>
      ) : null}

      {layer.type === "dense" || layer.type === "output" ? (
        <div className="layer-card-row">
          <label className="layer-card-label">Neurons</label>
          <NeuralInput
            type="number"
            min={1}
            max={512}
            value={layer.neurons}
            onChange={(e) => onChange({ neurons: parseInt(e.target.value) || 1 })}
            className="layer-card-input-control"
          />
        </div>
      ) : null}

      {layer.type === "conv2d" ? (
        <div className="layer-card-grid">
          <div className="layer-card-row">
            <label className="layer-card-label">Filters</label>
            <NeuralInput
              type="number"
              min={1}
              value={layer.filters ?? layer.neurons}
              onChange={(e) => {
                const val = parseInt(e.target.value) || 1;
                onChange({ filters: val, neurons: val });
              }}
              className="layer-card-input-control"
            />
          </div>
          <div className="layer-card-row">
            <label className="layer-card-label">Kernel</label>
            <NeuralInput
              type="number"
              min={1}
              value={layer.kernel_size ?? 3}
              onChange={(e) => onChange({ kernel_size: parseInt(e.target.value) || 3 })}
              className="layer-card-input-control"
            />
          </div>
          <div className="layer-card-row">
            <label className="layer-card-label">Stride</label>
            <NeuralInput
              type="number"
              min={1}
              value={layer.stride ?? 1}
              onChange={(e) => onChange({ stride: parseInt(e.target.value) || 1 })}
              className="layer-card-input-control"
            />
          </div>
          <div className="layer-card-row">
            <label className="layer-card-label">Padding</label>
            <NeuralSelect
              value={layer.padding ?? "valid"}
              onChange={(e) => onChange({ padding: e.target.value })}
              className="layer-card-input-control"
            >
              <option value="valid">valid</option>
              <option value="same">same</option>
            </NeuralSelect>
          </div>
        </div>
      ) : null}

      {layer.type === "maxpool2d" || layer.type === "avgpool2d" ? (
        <div className="layer-card-row">
          <label className="layer-card-label">Pool</label>
          <NeuralInput
            type="number"
            min={1}
            value={layer.pool_size ?? 2}
            onChange={(e) => onChange({ pool_size: parseInt(e.target.value) || 2 })}
            className="layer-card-input-control"
          />
          <label className="layer-card-label">Stride</label>
          <NeuralInput
            type="number"
            min={1}
            value={layer.pool_stride ?? layer.stride ?? layer.pool_size ?? 2}
            onChange={(e) => onChange({ pool_stride: parseInt(e.target.value) || 2 })}
            className="layer-card-input-control"
          />
        </div>
      ) : null}

      {layer.type === "embedding" ? (
        <div className="layer-card-row">
          <label className="layer-card-label">Vocab</label>
          <NeuralInput
            type="number"
            min={2}
            value={layer.vocab_size ?? 50}
            onChange={(e) => onChange({ vocab_size: parseInt(e.target.value) || 50 })}
            className="layer-card-input-control"
          />
          <label className="layer-card-label">Dim</label>
          <NeuralInput
            type="number"
            min={1}
            value={layer.embedding_dim ?? layer.neurons}
            onChange={(e) => onChange({ embedding_dim: parseInt(e.target.value) || 16, neurons: parseInt(e.target.value) || 16 })}
            className="layer-card-input-control"
          />
        </div>
      ) : null}

      {layer.type === "attention" ? (
        <div className="layer-card-row">
          <label className="layer-card-label">Heads</label>
          <NeuralInput
            type="number"
            min={1}
            value={layer.num_heads ?? 1}
            onChange={(e) => onChange({ num_heads: parseInt(e.target.value) || 1 })}
            className="layer-card-input-control"
          />
        </div>
      ) : null}

      {layer.type === "rnn" || layer.type === "lstm" || layer.type === "gru" ? (
        <div className="layer-card-row">
          <label className="layer-card-label">Hidden</label>
          <NeuralInput
            type="number"
            min={1}
            value={layer.hidden_size ?? layer.neurons}
            onChange={(e) => {
              const val = parseInt(e.target.value) || 1;
              onChange({ hidden_size: val, neurons: val });
            }}
            className="layer-card-input-control"
          />
          <label className="layer-card-label">Return Seq</label>
          <input
            type="checkbox"
            checked={Boolean(layer.return_sequences)}
            onChange={(e) => onChange({ return_sequences: e.target.checked })}
          />
        </div>
      ) : null}

      {!isInput && layer.type !== "embedding" && layer.type !== "attention" && layer.type !== "maxpool2d" && layer.type !== "avgpool2d" && layer.type !== "flatten" && layer.type !== "batchnorm" && layer.type !== "residual" ? (
        <div className="layer-card-row">
          <label className="layer-card-label">Activation</label>
          <NeuralSelect
            value={layer.activation ?? "linear"}
            onChange={(e) => onChange({ activation: e.target.value })}
            className="layer-card-input-control"
          >
            {activationOptions.map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </NeuralSelect>
        </div>
      ) : null}

      {!isInput && layer.type !== "maxpool2d" && layer.type !== "avgpool2d" && layer.type !== "flatten" && layer.type !== "batchnorm" && layer.type !== "residual" ? (
        <div className="layer-card-row">
          <label className="layer-card-label">Init</label>
          <NeuralSelect
            value={layer.init ?? "xavier"}
            onChange={(e) => onChange({ init: e.target.value })}
            className="layer-card-input-control"
          >
            {initOptions.map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </NeuralSelect>
        </div>
      ) : null}
    </div>
  );
}
