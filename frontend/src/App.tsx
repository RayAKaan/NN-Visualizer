import React, { useState } from "react";
import TopBar from "./components/layout/TopBar";
import StatusBar from "./components/layout/StatusBar";
import PredictionMode from "./components/prediction/PredictionMode";
import TrainingMode from "./components/training/TrainingMode";
import { useModelInfo } from "./hooks/useModelInfo";
import { useModelSwitcher } from "./hooks/useModelSwitcher";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const App: React.FC = () => {
  const [mode, setMode] = useState<"predict" | "train">("predict");
  const [view, setView] = useState<"2d" | "3d" | "compare">("2d");
  const [inferenceTime, setInferenceTime] = useState(0);
  const [canvasEmpty, setCanvasEmpty] = useState(true);

  const { activeModel, availableModels, switching, switchModel } = useModelSwitcher(API_BASE);
  const modelInfo = useModelInfo(API_BASE, activeModel);

  return (
    <div className="app-layout">
      <TopBar
        mode={mode}
        onModeChange={setMode}
        view={view}
        onViewChange={setView}
        modelInfo={modelInfo}
        activeModel={activeModel}
        availableModels={availableModels}
        switchingModel={switching}
        onSwitchModel={switchModel}
      />
      {mode === "predict" ? (
        <PredictionMode
          apiBase={API_BASE}
          view={view}
          activeModel={activeModel}
          canCompare={availableModels.includes("cnn")}
          onCanvasEmptyChange={setCanvasEmpty}
          onInferenceTimeChange={setInferenceTime}
        />
      ) : (
        <div className="main-content train-content">
          <TrainingMode />
        </div>
      )}
      <StatusBar
        connected={Boolean(modelInfo)}
        modelType={activeModel}
        inferenceTime={inferenceTime}
        isCanvasEmpty={canvasEmpty}
      />
    </div>
  );
};

export default App;
