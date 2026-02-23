import React from "react";
import CanvasGrid from "./CanvasGrid";
import ConnectionView from "./ConnectionView";
import LayerView from "./LayerView";
import Network3D from "./Network3D";
import { NeuralState } from "../types/NeuralState";

interface PredictionModeProps {
  view: "2d" | "3d";
  onToggleView: () => void;
  probabilities: number[];
  maxIndex: number;
  hidden1: number[];
  hidden2: number[];
  output: number[];
  state3D: NeuralState | null;
  weightsHidden1Hidden2: number[][] | null;
  weightsHidden2Output: number[][] | null;
  pixels: number[];
  onPixelsChange: (next: number[]) => void;
  prediction: number | null;
  error: string | null;
}

const PredictionMode: React.FC<PredictionModeProps> = ({
  view,
  onToggleView,
  probabilities,
  maxIndex,
  hidden1,
  hidden2,
  output,
  state3D,
  weightsHidden1Hidden2,
  weightsHidden2Output,
  pixels,
  onPixelsChange,
  prediction,
  error,
}) => {
  return (
    <section className="visualizer">
      <div className="view-toggle">
        <button type="button" onClick={onToggleView}>
          Switch to {view === "2d" ? "3D" : "2D"}
        </button>
      </div>
      <div className="output-row">
        {probabilities.map((value, index) => (
          <div key={`digit-${index}`} className="digit-cell">
            <span className="digit-label">{index}</span>
            <div className={`digit-box ${index === maxIndex ? "active" : ""}`}>
              <div className="digit-fill" style={{ opacity: Math.min(1, value * 2 + 0.1) }} />
            </div>
          </div>
        ))}
      </div>

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
            <LayerView title="Hidden 1" activations={hidden1} columns={32} showTitle={false} />
          </div>
          <div className="layer-row">
            <LayerView title="Hidden 2" activations={hidden2} columns={16} showTitle={false} />
          </div>
        </div>
      ) : (
        <div className="network-stage">{state3D ? <Network3D state={state3D} /> : <p>Loading 3D scene…</p>}</div>
      )}

      <div className="input-panel">
        <CanvasGrid pixels={pixels} onChange={onPixelsChange} />
        <button type="button" className="clear-button" onClick={() => onPixelsChange(Array.from({ length: 784 }, () => 0))}>
          Clear
        </button>
        {prediction !== null && <p className="prediction">Prediction: {prediction}</p>}
        {error && <p className="error">{error}</p>}
      </div>
    </section>
  );
};

export default PredictionMode;
