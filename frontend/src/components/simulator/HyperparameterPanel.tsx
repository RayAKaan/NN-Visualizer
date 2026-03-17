import React from "react";
import { useTrainingSimStore } from "../../store/trainingSimStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralSelect } from "@/design-system/components/NeuralSelect";

export function HyperparameterPanel() {
  const config = useTrainingSimStore((s) => s.config);
  const updateConfig = useTrainingSimStore((s) => s.updateConfig);

  return (
    <NeuralPanel className="hyper-panel" variant="base">
      <div className="hyper-title">Training Configuration</div>
      <label className="hyper-label">
        Epochs
        <NeuralInput
          type="number"
          value={config.epochs}
          onChange={(e) => updateConfig({ epochs: parseInt(e.target.value) || 1 })}
          className="hyper-control"
        />
      </label>
      <label className="hyper-label">
        Batch Size
        <NeuralInput
          type="number"
          value={config.batch_size}
          onChange={(e) => updateConfig({ batch_size: parseInt(e.target.value) || 1 })}
          className="hyper-control"
        />
      </label>
      <label className="hyper-label">
        Learning Rate
        <NeuralInput
          type="number"
          step="0.0001"
          value={config.learning_rate}
          onChange={(e) => updateConfig({ learning_rate: parseFloat(e.target.value) || 0.001 })}
          className="hyper-control"
        />
      </label>
      <label className="hyper-label">
        Optimizer
        <NeuralSelect
          value={config.optimizer}
          onChange={(e) => updateConfig({ optimizer: e.target.value })}
          className="hyper-control"
        >
          <option value="adam">Adam</option>
          <option value="sgd">SGD</option>
          <option value="sgd_momentum">SGD Momentum</option>
          <option value="rmsprop">RMSProp</option>
        </NeuralSelect>
      </label>
      <label className="hyper-label">
        Loss Function
        <NeuralSelect
          value={config.loss_function}
          onChange={(e) => updateConfig({ loss_function: e.target.value })}
          className="hyper-control"
        >
          <option value="bce">Binary Cross-Entropy</option>
          <option value="mse">MSE</option>
        </NeuralSelect>
      </label>
      <label className="hyper-label">
        L2 Lambda
        <NeuralInput
          type="number"
          step="0.0001"
          value={config.l2_lambda}
          onChange={(e) => updateConfig({ l2_lambda: parseFloat(e.target.value) || 0 })}
          className="hyper-control"
        />
      </label>
      <label className="hyper-label">
        Dropout
        <NeuralInput
          type="number"
          step="0.05"
          value={config.dropout_rate}
          onChange={(e) => updateConfig({ dropout_rate: parseFloat(e.target.value) || 0 })}
          className="hyper-control"
        />
      </label>
    </NeuralPanel>
  );
}
