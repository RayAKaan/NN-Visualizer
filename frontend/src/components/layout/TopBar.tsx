import React from "react";
import { ModelInfo } from "../../types";

interface Props {
  mode: "predict" | "train";
  onModeChange: (mode: "predict" | "train") => void;
  view: "2d" | "3d";
  onViewChange: (view: "2d" | "3d") => void;
  modelInfo: ModelInfo | null;
}

const TopBar: React.FC<Props> = ({ mode, onModeChange, view, onViewChange, modelInfo }) => {
  return (
    <header className="top-bar">
      <div className="brand">Neural Network Visualizer</div>
      <div className="toggle-group">
        <button className={`toggle-option ${mode === "predict" ? "active" : ""}`} onClick={() => onModeChange("predict")}>Predict</button>
        <button className={`toggle-option ${mode === "train" ? "active" : ""}`} onClick={() => onModeChange("train")}>Train</button>
      </div>
      <div className="toggle-group">
        <button className={`toggle-option ${view === "2d" ? "active" : ""}`} onClick={() => onViewChange("2d")}>2D</button>
        <button className={`toggle-option ${view === "3d" ? "active" : ""}`} onClick={() => onViewChange("3d")}>3D</button>
      </div>
      <div className="model-badge">
        {modelInfo
          ? `${modelInfo.type.toUpperCase()} · 784→128→64→10 · ${(modelInfo.accuracy * 100).toFixed(1)}%`
          : "Model loading…"}
      </div>
    </header>
  );
};

export default TopBar;
