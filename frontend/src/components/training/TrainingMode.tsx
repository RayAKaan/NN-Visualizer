import React from "react";
import TrainingControls from "./TrainingControls";
import TrainingChart from "./TrainingChart";
import ConnectionView from "../ConnectionView";
import LayerView from "../LayerView";
import Network3D from "../Network3D";
import { NeuralState } from "../../types/NeuralState";

interface TrainingModeProps {
  view: "2d" | "3d";
  onToggleView: () => void;

  state3D: NeuralState | null;

  hidden1: number[];
  hidden2: number[];
  output: number[];

  weightsHidden1Hidden2: number[][] | null;
  weightsHidden2Output: number[][] | null;

  lossHistory: number[];
  accuracyHistory: number[];
  valLossHistory: number[];
  valAccuracyHistory: number[];

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
  lossHistory,
  accuracyHistory,
  valLossHistory,
  valAccuracyHistory,
  controlState,
  onControlChange,
  onCommand,
}) => {
  return (
    <section className="training-mode">
      {/* ================= View toggle ================= */}
      <div className="view-toggle">
        <button type="button" onClick={onToggleView}>
          Switch to {view === "2d" ? "3D" : "2D"}
        </button>
      </div>

      <div className="training-grid">
        {/* ================= Controls ================= */}
        <div className="training-panel">
          <TrainingControls
            {...controlState}
            onChange={onControlChange}
            onCommand={onCommand}
          />
        </div>

        {/* ================= Network ================= */}
        <div className="training-panel">
          {view === "2d" ? (
            <div className="network-stage">
              <ConnectionView
                hidden1={hidden1}
                hidden2={hidden2}
                output={output}
                weightsHidden1Hidden2={weightsHidden1Hidden2}
                weightsHidden2Output={weightsHidden2Output}
                width={720}
                height={220}
              />

              <div className="layer-row">
                <LayerView
                  title="Hidden Layer 1"
                  activations={hidden1}
                  columns={32}
                  showTitle={false}
                />
              </div>

              <div className="layer-row">
                <LayerView
                  title="Hidden Layer 2"
                  activations={hidden2}
                  columns={16}
                  showTitle={false}
                />
              </div>
            </div>
          ) : (
            <div className="network-stage">
              {state3D ? (
                <Network3D
                  hidden1={state3D.layers.hidden1}
                  hidden2={state3D.layers.hidden2}
                  output={state3D.layers.output}
                  weightsHidden1Hidden2={
                    state3D.edges.hidden1_hidden2.map(e => [e.strength])
                  }
                  weightsHidden2Output={
                    state3D.edges.hidden2_output.map(e => [e.strength])
                  }
                />
              ) : (
                <p className="loading">Loading 3D scene…</p>
              )}
            </div>
          )}
        </div>

        {/* ================= Charts ================= */}
        <div className="training-panel charts">
          <TrainingChart title="Loss" values={lossHistory} />
          <TrainingChart title="Accuracy" values={accuracyHistory} color="#0f766e" />
          <TrainingChart title="Val Loss" values={valLossHistory} color="#f97316" />
          <TrainingChart title="Val Accuracy" values={valAccuracyHistory} color="#2563eb" />
        </div>
      </div>
    </section>
  );
};

export default TrainingMode;
