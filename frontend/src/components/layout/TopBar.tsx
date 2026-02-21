import React from "react";
import { ModelInfo, ModelType } from "../../types";
import ModelSwitcher from "./ModelSwitcher";

interface Props {
  mode: "predict" | "train";
  onModeChange: (mode: "predict" | "train") => void;
  view: "2d" | "3d" | "compare";
  onViewChange: (view: "2d" | "3d" | "compare") => void;
  modelInfo: ModelInfo | null;
  activeModel: ModelType;
  availableModels: ModelType[];
  switchingModel: boolean;
  onSwitchModel: (type: ModelType) => void;
}

const TopBar: React.FC<Props> = ({
  mode,
  onModeChange,
  view,
  onViewChange,
  modelInfo,
  activeModel,
  availableModels,
  switchingModel,
  onSwitchModel,
}) => {
  const modelSummary =
    activeModel === "cnn"
      ? "CNN · Conv32→Pool→Conv64→Pool→Dense128→10"
      : "ANN · 784→128→64→10";

  return (
    <header className="top-bar">
      <div className="brand">Neural Network Visualizer</div>
      <div className="toggle-group">
        <button className={`toggle-option ${mode === "predict" ? "active" : ""}`} onClick={() => onModeChange("predict")}>Predict</button>
        <button className={`toggle-option ${mode === "train" ? "active" : ""}`} onClick={() => onModeChange("train")}>Train</button>
      </div>

      <ModelSwitcher active={activeModel} available={availableModels} switching={switchingModel} onSwitch={onSwitchModel} />

      <div className="toggle-group">
        <button className={`toggle-option ${view === "2d" ? "active" : ""}`} onClick={() => onViewChange("2d")}>2D</button>
        <button className={`toggle-option ${view === "3d" ? "active" : ""}`} onClick={() => onViewChange("3d")}>3D</button>
        <button className={`toggle-option ${view === "compare" ? "active" : ""}`} onClick={() => onViewChange("compare")} disabled={!availableModels.includes("cnn")}>Compare</button>
      </div>
      <div className="model-badge">
        {modelInfo ? modelSummary : "Model loading…"}
      </div>
    </header>
  );
};

export default TopBar;
