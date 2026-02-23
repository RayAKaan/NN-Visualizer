import React, { useState, useEffect } from "react";
import { TrainingStatus, TrainingConfig } from "../../types";

interface Props {
  status: TrainingStatus;
  config: TrainingConfig;
  onConfigChange: (config: Partial<TrainingConfig>) => void;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  onStepBatch: () => void;
  onStepEpoch: () => void;
  currentEpoch: number;
  currentBatch: number;
  totalBatches: number;
  totalEpochs: number;
}

export default function TrainingControls({
  status,
  config,
  onConfigChange,
  onStart,
  onPause,
  onResume,
  onStop,
  onStepBatch,
  onStepEpoch,
  currentEpoch,
  currentBatch,
  totalBatches,
  totalEpochs,
}: Props) {
  const [localConfig, setLocalConfig] = useState<TrainingConfig>({ ...config });
  const isIdle = status === "idle" || status === "completed" || status === "error";
  const isRunning = status === "running";
  const isPaused = status === "paused";

  useEffect(() => {
    if (isIdle) setLocalConfig({ ...config });
  }, [config, status, isIdle]);

  const updateConfig = (key: string, value: any) => {
    const updated = { ...localConfig, [key]: value };
    setLocalConfig(updated);
    onConfigChange({ [key]: value });
  };

  return (
    <div className="training-controls">
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span className={`status-badge ${status}`}>
          <span
            className={`status-dot ${status === "running" ? "connected" : status === "error" ? "disconnected" : ""}`}
            style={{ width: 6, height: 6 }}
          />
          {status.toUpperCase()}
        </span>
      </div>

      {(isRunning || isPaused) && (
        <div className="card" style={{ padding: "8px 12px" }}>
          <div className="info-row">
            <span>Epoch</span>
            <span>
              {currentEpoch} / {totalEpochs}
            </span>
          </div>
          <div className="info-row">
            <span>Batch</span>
            <span>
              {currentBatch} / {totalBatches}
            </span>
          </div>
          <div
            style={{
              height: 4,
              background: "var(--bg-secondary)",
              borderRadius: 2,
              marginTop: 6,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${totalBatches > 0 ? (currentBatch / totalBatches) * 100 : 0}%`,
                background: "var(--accent-cyan)",
                borderRadius: 2,
                transition: "width 100ms",
              }}
            />
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">Hyperparameters</div>

        <div className="control-group">
          <label className="control-label">Model</label>
          <select
            className="control-input"
            value={localConfig.model_type}
            onChange={(e) => updateConfig("model_type", e.target.value)}
            disabled={!isIdle}
          >
            <option value="ann">ANN (Dense)</option>
            <option value="cnn">CNN (Conv)</option>
          </select>
        </div>

        <div className="control-group">
          <label className="control-label">Learning Rate: {localConfig.learning_rate.toFixed(4)}</label>
          <input
            type="range"
            className="control-slider"
            min={-4}
            max={-1}
            step={0.1}
            value={Math.log10(localConfig.learning_rate)}
            onChange={(e) =>
              updateConfig("learning_rate", parseFloat(Math.pow(10, parseFloat(e.target.value)).toFixed(6)))
            }
            disabled={!isIdle}
          />
        </div>

        <div className="control-group">
          <label className="control-label">Batch Size</label>
          <select
            className="control-input"
            value={localConfig.batch_size}
            onChange={(e) => updateConfig("batch_size", parseInt(e.target.value, 10))}
            disabled={!isIdle}
          >
            {[16, 32, 64, 128, 256].map((bs) => (
              <option key={bs} value={bs}>
                {bs}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label className="control-label">Epochs</label>
          <input
            type="number"
            className="control-input"
            min={1}
            max={100}
            value={localConfig.epochs}
            onChange={(e) => updateConfig("epochs", parseInt(e.target.value, 10) || 1)}
            disabled={!isIdle}
          />
        </div>

        <div className="control-group">
          <label className="control-label">Optimizer</label>
          <select
            className="control-input"
            value={localConfig.optimizer}
            onChange={(e) => updateConfig("optimizer", e.target.value)}
            disabled={!isIdle}
          >
            <option value="adam">Adam</option>
            <option value="sgd">SGD (momentum=0.9)</option>
            <option value="rmsprop">RMSprop</option>
          </select>
        </div>

        <div className="control-group">
          <label className="control-label">Activation</label>
          <select
            className="control-input"
            value={localConfig.activation}
            onChange={(e) => updateConfig("activation", e.target.value)}
            disabled={!isIdle}
          >
            <option value="relu">ReLU</option>
            <option value="tanh">Tanh</option>
            <option value="sigmoid">Sigmoid</option>
          </select>
        </div>

        <div className="control-group">
          <label className="control-label">Dropout: {localConfig.dropout_rate.toFixed(2)}</label>
          <input
            type="range"
            className="control-slider"
            min={0}
            max={0.5}
            step={0.05}
            value={localConfig.dropout_rate}
            onChange={(e) => updateConfig("dropout_rate", parseFloat(e.target.value))}
            disabled={!isIdle}
          />
        </div>
      </div>

      <div className="card">
        <div className="card-title">Controls</div>
        <div className="runtime-buttons">
          {isIdle && (
            <button className="btn btn-start btn-block" onClick={onStart} style={{ gridColumn: "1 / -1" }}>
              ▶ Start Training
            </button>
          )}
          {isRunning && (
            <>
              <button className="btn btn-pause" onClick={onPause}>
                ⏸ Pause
              </button>
              <button className="btn btn-stop" onClick={onStop}>
                ⏹ Stop
              </button>
            </>
          )}
          {isPaused && (
            <>
              <button className="btn btn-start" onClick={onResume}>
                ▶ Resume
              </button>
              <button className="btn btn-stop" onClick={onStop}>
                ⏹ Stop
              </button>
              <button className="btn btn-step" onClick={onStepBatch}>
                ⏭ Step Batch
              </button>
              <button className="btn btn-step" onClick={onStepEpoch}>
                ⏩ Step Epoch
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
