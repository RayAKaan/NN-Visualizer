import React, { useEffect, useState, useCallback } from "react";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";
import { useTrainingHistory } from "../../hooks/useTrainingHistory";
import TrainingControls from "./TrainingControls";
import TrainingChart from "./TrainingChart";
import TrainingNetworkView from "./TrainingNetworkView";
import TrainingReplay from "./TrainingReplay";
import EpochTable from "./EpochTable";
import { BatchUpdate, TrainingConfig, LayerActivation, GradientInfo } from "../../types";

export default function TrainingMode() {
  const {
    connected,
    connect,
    disconnect,
    status,
    currentEpoch,
    currentBatch,
    totalBatches,
    totalEpochs,
    error,
    completionInfo,
    latestBatch,
    metrics,
    epochSummaries,
    configure,
    start,
    pause,
    resume,
    stop,
    stepBatch,
    stepEpoch,
  } = useTrainingSocket();

  const { history, fetchHistory } = useTrainingHistory();

  const [config, setConfig] = useState<TrainingConfig>({
    model_type: "ann",
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 5,
    optimizer: "adam",
    activation: "relu",
    weight_decay: 0,
    dropout_rate: 0,
    kernel_initializer: "glorot_uniform",
  });

  const [replayActive, setReplayActive] = useState(false);
  const [replayActivations, setReplayActivations] = useState<Record<string, LayerActivation> | null>(null);
  const [replayGradients, setReplayGradients] = useState<Record<string, GradientInfo> | null>(null);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  useEffect(() => {
    if (status === "completed" || status === "idle") {
      fetchHistory();
    }
  }, [status, fetchHistory]);

  const handleConfigChange = useCallback(
    (partial: Partial<TrainingConfig>) => {
      setConfig((prev) => ({ ...prev, ...partial }));
      if (connected) {
        configure(partial);
      }
    },
    [connected, configure]
  );

  const handleStart = useCallback(() => {
    if (connected) {
      configure(config);
      setTimeout(() => start(), 100);
    }
  }, [connected, config, configure, start]);

  const handleReplaySnapshot = useCallback((snapshot: BatchUpdate) => {
    setReplayActive(true);
    setReplayActivations(snapshot.activations as Record<string, LayerActivation>);
    setReplayGradients(snapshot.gradients as Record<string, GradientInfo>);
  }, []);

  const displayActivations = replayActive
    ? replayActivations
    : (latestBatch?.activations as Record<string, LayerActivation> | null) ?? null;
  const displayGradients = replayActive
    ? replayGradients
    : (latestBatch?.gradients as Record<string, GradientInfo> | null) ?? null;

  useEffect(() => {
    if (status === "running") {
      setReplayActive(false);
    }
  }, [status]);

  const epochOverlayPoints = metrics.epochBoundaries.map((batchIdx, i) => ({
    index: batchIdx,
    value: metrics.valLosses[i] ?? 0,
    color: "#f59e0b",
  }));

  return (
    <>
      <div className="left-panel">
        {!connected && (
          <div className="card" style={{ borderColor: "var(--accent-red)" }}>
            <div style={{ fontSize: 12, color: "var(--accent-red)", marginBottom: 8 }}>
              ⚠ Not connected to training server
            </div>
            <button className="btn btn-sm btn-primary" onClick={connect}>
              Reconnect
            </button>
          </div>
        )}

        {error && (
          <div className="card" style={{ borderColor: "var(--accent-red)" }}>
            <div style={{ fontSize: 11, color: "var(--accent-red)", fontFamily: "var(--font-mono)" }}>
              Error: {error}
            </div>
          </div>
        )}

        {completionInfo && (
          <div className="card" style={{ borderColor: "var(--accent-green)" }}>
            <div className="card-title" style={{ color: "var(--accent-green)" }}>
              ✓ Training Complete
            </div>
            <div className="info-row">
              <span>Epochs</span>
              <span>{completionInfo.epochs}</span>
            </div>
            <div className="info-row">
              <span>Final Accuracy</span>
              <span style={{ color: "var(--accent-green)", fontWeight: 600 }}>
                {(completionInfo.finalAccuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="info-row">
              <span>Snapshots</span>
              <span>{completionInfo.snapshots}</span>
            </div>
          </div>
        )}

        <TrainingControls
          status={status}
          config={config}
          onConfigChange={handleConfigChange}
          onStart={handleStart}
          onPause={pause}
          onResume={resume}
          onStop={stop}
          onStepBatch={stepBatch}
          onStepEpoch={stepEpoch}
          currentEpoch={currentEpoch}
          currentBatch={currentBatch}
          totalBatches={totalBatches}
          totalEpochs={totalEpochs}
        />
      </div>

      <div className="training-center">
        <div className="training-network">
          <TrainingNetworkView
            activations={displayActivations}
            gradients={displayGradients}
            modelType={config.model_type}
          />
        </div>

        {(status === "completed" || status === "idle") && history.length > 0 && (
          <TrainingReplay history={history} onSnapshotChange={handleReplaySnapshot} />
        )}
      </div>

      <div className="right-panel">
        <TrainingChart
          data={metrics.losses}
          label="Loss"
          color="#ef4444"
          height={110}
          overlayPoints={epochOverlayPoints}
        />

        <TrainingChart
          data={metrics.accuracies}
          label="Accuracy"
          color="#10b981"
          height={110}
          minY={0}
          maxY={1}
        />

        <TrainingChart data={metrics.gradientNorms} label="Gradient Norm" color="#f59e0b" height={90} />

        {latestBatch && (
          <div className="card" style={{ padding: "8px 12px" }}>
            <div className="info-row">
              <span>Loss</span>
              <span style={{ color: "var(--accent-red)" }}>{latestBatch.loss.toFixed(4)}</span>
            </div>
            <div className="info-row">
              <span>Accuracy</span>
              <span style={{ color: "var(--accent-green)" }}>{(latestBatch.accuracy * 100).toFixed(1)}%</span>
            </div>
            <div className="info-row">
              <span>Grad Norm</span>
              <span style={{ color: "var(--accent-amber)" }}>{latestBatch.gradient_norm.toFixed(4)}</span>
            </div>
            <div className="info-row">
              <span>LR</span>
              <span>{latestBatch.learning_rate.toFixed(5)}</span>
            </div>
          </div>
        )}

        <EpochTable epochs={epochSummaries} />
      </div>
    </>
  );
}
