import React from "react";

interface TrainingControlsProps {
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: string;
  weightDecay: number;
  activation: string;
  initializer: string;
  dropout: number;
  onChange: (field: string, value: number | string) => void;
  onCommand: (command: string) => void;
}

const TrainingControls: React.FC<TrainingControlsProps> = ({
  learningRate,
  batchSize,
  epochs,
  optimizer,
  weightDecay,
  activation,
  initializer,
  dropout,
  onChange,
  onCommand,
}) => {
  return (
    <div className="training-controls">
      <h3>Training controls</h3>
      <div className="control-grid">
        <label>
          Learning rate
          <input
            type="number"
            step="0.0001"
            value={learningRate}
            onChange={(event) => onChange("learningRate", Number(event.target.value))}
          />
        </label>
        <label>
          Batch size
          <select value={batchSize} onChange={(event) => onChange("batchSize", Number(event.target.value))}>
            {[32, 64, 128, 256].map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </label>
        <label>
          Epochs
          <input
            type="number"
            value={epochs}
            min={1}
            onChange={(event) => onChange("epochs", Number(event.target.value))}
          />
        </label>
        <label>
          Optimizer
          <select value={optimizer} onChange={(event) => onChange("optimizer", event.target.value)}>
            {"adam,sgd,rmsprop".split(",").map((opt) => (
              <option key={opt} value={opt}>
                {opt.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
        <label>
          Weight decay
          <input
            type="number"
            step="0.0001"
            value={weightDecay}
            onChange={(event) => onChange("weightDecay", Number(event.target.value))}
          />
        </label>
        <label>
          Activation
          <select value={activation} onChange={(event) => onChange("activation", event.target.value)}>
            {["relu", "leaky_relu"].map((act) => (
              <option key={act} value={act}>
                {act}
              </option>
            ))}
          </select>
        </label>
        <label>
          Initializer
          <select value={initializer} onChange={(event) => onChange("initializer", event.target.value)}>
            {["glorot_uniform", "he_normal"].map((init) => (
              <option key={init} value={init}>
                {init}
              </option>
            ))}
          </select>
        </label>
        <label>
          Dropout
          <input
            type="number"
            step="0.05"
            min={0}
            max={0.8}
            value={dropout}
            onChange={(event) => onChange("dropout", Number(event.target.value))}
          />
        </label>
      </div>
      <div className="control-buttons">
        <button type="button" onClick={() => onCommand("configure")}>Configure</button>
        <button type="button" onClick={() => onCommand("start")}>Start</button>
        <button type="button" onClick={() => onCommand("pause")}>Pause</button>
        <button type="button" onClick={() => onCommand("resume")}>Resume</button>
        <button type="button" onClick={() => onCommand("stop")}>Stop</button>
        <button type="button" onClick={() => onCommand("step_batch")}>Step batch</button>
        <button type="button" onClick={() => onCommand("step_epoch")}>Step epoch</button>
      </div>
    </div>
  );
};

export default TrainingControls;
