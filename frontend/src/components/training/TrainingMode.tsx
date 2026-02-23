import { useEffect, useMemo, useState } from "react";
import { ModelType, TrainingConfig } from "../../types";
import { useTrainingSocket } from "../../hooks/useTrainingSocket";
import ConfusionMatrix from "./ConfusionMatrix";
import EpochTable from "./EpochTable";
import MetricsDashboard from "./MetricsDashboard";
import TrainingChart from "./TrainingChart";
import TrainingControls from "./TrainingControls";
import TrainingNetworkView from "./TrainingNetworkView";
import TrainingReplay from "./TrainingReplay";

type Props = {
  trainingToggleSignal?: number;
  availableModels?: ModelType[];
};

export default function TrainingMode({ trainingToggleSignal = 0, availableModels = ["ann", "cnn", "rnn"] }: Props) {
  const { connected, status, metrics, connect, configure, start, pause, resume, stop } = useTrainingSocket();
  const [modelType, setModelType] = useState<ModelType>(availableModels[0] ?? "ann");
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(64);
  const [epochs, setEpochs] = useState(5);
  const [optimizer, setOptimizer] = useState("adam");

  const running = status === "running";
  const paused = status === "paused";

  const config: TrainingConfig = useMemo(() => ({
    model_type: modelType,
    learning_rate: learningRate,
    batch_size: batchSize,
    epochs,
    optimizer,
    activation: "relu",
    dropout_rate: 0.2,
    kernel_initializer: "glorot_uniform",
    lstm_units: 128,
    bidirectional: false,
    conv1_filters: 32,
    conv2_filters: 64,
  }), [modelType, learningRate, batchSize, epochs, optimizer]);

  useEffect(() => {
    if (!connected) connect();
  }, [connected, connect]);

  useEffect(() => {
    if (!trainingToggleSignal) return;
    if (running) pause();
    else if (paused) resume();
  }, [trainingToggleSignal, running, paused, pause, resume]);

  const onStart = () => {
    configure(config);
    start();
  };

  const latestLoss = metrics.losses[metrics.losses.length - 1] ?? 0;
  const latestAcc = metrics.accuracies[metrics.accuracies.length - 1] ?? 0;
  const latestValLoss = metrics.valLosses[metrics.valLosses.length - 1] ?? 0;
  const latestValAcc = metrics.valAccuracies[metrics.valAccuracies.length - 1] ?? 0;
  const latestGradNorm = metrics.gradientNorms[metrics.gradientNorms.length - 1] ?? 0;
  const latestLr = metrics.learningRates[metrics.learningRates.length - 1] ?? learningRate;
  const lastConfusion = metrics.confusionMatrices[metrics.confusionMatrices.length - 1] ?? null;

  return (
    <>
      <div className="left-panel">
        <TrainingControls connected={connected} status={status} onConnect={connect} onStart={onStart} onPause={pause} onResume={resume} onStop={stop} />
        <div className="card" style={{ marginTop: 12 }}>
          <h3>⚙️ Hyperparameters</h3>
          <div className="control-group">
            <label className="control-label">Model</label>
            <select className="control-input" value={modelType} onChange={(e) => setModelType(e.target.value as ModelType)} disabled={running || paused}>
              {availableModels.map((m) => <option key={m} value={m}>{m.toUpperCase()}</option>)}
            </select>
            <label className="control-label">Learning Rate: {learningRate.toFixed(4)}</label>
            <input className="control-slider" type="range" min="0.0001" max="0.01" step="0.0001" value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))} disabled={running || paused} />
            <label className="control-label">Batch Size: {batchSize}</label>
            <input className="control-slider" type="range" min="16" max="256" step="16" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} disabled={running || paused} />
            <label className="control-label">Epochs: {epochs}</label>
            <input className="control-slider" type="range" min="1" max="30" step="1" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} disabled={running || paused} />
            <label className="control-label">Optimizer</label>
            <select className="control-input" value={optimizer} onChange={(e) => setOptimizer(e.target.value)} disabled={running || paused}>
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
              <option value="rmsprop">RMSprop</option>
            </select>
          </div>
        </div>
      </div>

      <div className="center-stage training-center">
        <TrainingChart losses={metrics.losses} accuracies={metrics.accuracies} valLosses={metrics.valLosses} valAccuracies={metrics.valAccuracies} />
        <div style={{ marginTop: 12 }}>
          <EpochTable losses={metrics.losses} accuracies={metrics.accuracies} valLosses={metrics.valLosses} valAccuracies={metrics.valAccuracies} />
        </div>
        <div style={{ marginTop: 12 }}>
          <ConfusionMatrix matrix={lastConfusion} />
        </div>
      </div>

      <div className="right-panel">
        <MetricsDashboard latestLoss={latestLoss} latestAccuracy={latestAcc} latestValLoss={latestValLoss} latestValAccuracy={latestValAcc} latestGradientNorm={latestGradNorm} latestLearningRate={latestLr} />
        <div style={{ marginTop: 12 }}>
          <TrainingNetworkView gradients={metrics.gradientNorms} />
        </div>
        <div style={{ marginTop: 12 }}>
          <TrainingReplay losses={metrics.losses} accuracies={metrics.accuracies} />
        </div>
      </div>
    </>
  );
}
