import React from "react";
import TrainingControls from "./TrainingControls";
import TrainingChart from "./TrainingChart";
import TrainingTimeline from "./TrainingTimeline";
import TrainingReplay from "./TrainingReplay";
import LossLandscape3D from "./LossLandscape3D";
import ConnectionView from "../ConnectionView";
import LayerView from "../LayerView";
import Network3D from "../Network3D";
import { NeuralState } from "../../types/NeuralState";
import { TrainingHistorySnapshot, useTrainingHistory } from "../../hooks/useTrainingHistory";

interface TrainingModeProps {
  view: "2d" | "3d";
  onToggleView: () => void;
  state3D: NeuralState | null;
  hidden1: number[];
  hidden2: number[];
  output: number[];
  weightsHidden1Hidden2: number[][] | null;
  weightsHidden2Output: number[][] | null;
  gradientsHidden1Hidden2?: number[][] | null;
  gradientsHidden2Output?: number[][] | null;
  lossHistory: number[];
  accuracyHistory: number[];
  valLossHistory: number[];
  valAccuracyHistory: number[];
  gradientNormHistory: number[];
  learningRateHistory: number[];
  weightHistory: number[];
  timelineIndex: number;
  timelineMax: number;
  onTimelineChange: (next: number) => void;
  onTimelineReplay: () => void;
  onReplaySnapshot: (snapshot: TrainingHistorySnapshot) => void;
  controlState: {
    learningRate: number;
    batchSize: number;
    epochs: number;
    optimizer: string;
    weightDecay: number;
    activation: string;
    initializer: string;
    dropout: number;
  };
  onControlChange: (field: string, value: number | string) => void;
  onCommand: (command: string) => void;
}

const TrainingMode: React.FC<TrainingModeProps> = ({
  view,
  onToggleView,
  state3D,
  hidden1,
  hidden2,
  output,
  weightsHidden1Hidden2,
  weightsHidden2Output,
  gradientsHidden1Hidden2 = null,
  gradientsHidden2Output = null,
  lossHistory,
  accuracyHistory,
  valLossHistory,
  valAccuracyHistory,
  gradientNormHistory,
  learningRateHistory,
  weightHistory,
  timelineIndex,
  timelineMax,
  onTimelineChange,
  onTimelineReplay,
  onReplaySnapshot,
  controlState,
  onControlChange,
  onCommand,
}) => {
  const { history, loaded } = useTrainingHistory();

  return (
    <section className="training-mode">
      <div className="view-toggle">
        <button type="button" onClick={onToggleView}>
          Switch to {view === "2d" ? "3D" : "2D"}
        </button>
      </div>

      <TrainingTimeline
        value={timelineIndex}
        max={timelineMax}
        onChange={onTimelineChange}
        onReplay={onTimelineReplay}
      />

      <TrainingReplay history={history} onSnapshotChange={onReplaySnapshot} />
      {!loaded && <p>Loading server training history…</p>}

      <div className="training-grid">
        <div className="training-panel">
          <TrainingControls {...controlState} onChange={onControlChange} onCommand={onCommand} />
        </div>
        <div className="training-panel">
          {view === "2d" ? (
            <div className="network-stage">
              <ConnectionView
                hidden1={hidden1}
                hidden2={hidden2}
                output={output}
                weightsHidden1Hidden2={weightsHidden1Hidden2}
                weightsHidden2Output={weightsHidden2Output}
                gradientsHidden1Hidden2={gradientsHidden1Hidden2}
                gradientsHidden2Output={gradientsHidden2Output}
                width={720}
                height={220}
              />
              <div className="layer-row">
                <LayerView title="Hidden 1" activations={hidden1} columns={32} showTitle={false} />
              </div>
              <div className="layer-row">
                <LayerView title="Hidden 2" activations={hidden2} columns={16} showTitle={false} />
              </div>
            </div>
          ) : (
            <div className="network-stage">
              {state3D ? <Network3D state={state3D} /> : <p>Loading 3D scene…</p>}
            </div>
          )}
        </div>
        <div className="training-panel charts">
          <TrainingChart title="Loss" values={lossHistory} />
          <TrainingChart title="Accuracy" values={accuracyHistory} color="#0f766e" />
          <TrainingChart title="Val Loss" values={valLossHistory} color="#f97316" />
          <TrainingChart title="Val Accuracy" values={valAccuracyHistory} color="#2563eb" />
          <TrainingChart title="Gradient norm" values={gradientNormHistory} color="#7c3aed" />
          <TrainingChart title="Learning rate" values={learningRateHistory} color="#0ea5e9" />
          <LossLandscape3D lossHistory={lossHistory} weightHistory={weightHistory} />
        </div>
      </div>
    </section>
  );
};

export default TrainingMode;
