import React, { useState } from "react";
import TopBar from "./components/layout/TopBar";
import StatusBar from "./components/layout/StatusBar";
import PredictionMode from "./components/prediction/PredictionMode";
import TrainingShell from "./components/training/TrainingShell";
import { useModelInfo } from "./hooks/useModelInfo";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const App: React.FC = () => {
  const [mode, setMode] = useState<"predict" | "train">("predict");
  const [view, setView] = useState<"2d" | "3d">("2d");
  const [inferenceTime, setInferenceTime] = useState(0);
  const [canvasEmpty, setCanvasEmpty] = useState(true);
  const modelInfo = useModelInfo(API_BASE);

  return (
    <div className="app-layout">
      <TopBar mode={mode} onModeChange={setMode} view={view} onViewChange={setView} modelInfo={modelInfo} />
      {mode === "predict" ? (
        <PredictionMode
          apiBase={API_BASE}
          view={view}
          onCanvasEmptyChange={setCanvasEmpty}
          onInferenceTimeChange={setInferenceTime}
        />
      ) : (
        <div className="main-content train-content">
          <TrainingShell apiBase={API_BASE} view={view} onToggleView={() => setView((v) => (v === "2d" ? "3d" : "2d"))} />
        </div>
      )}
      <StatusBar connected={Boolean(modelInfo)} modelType={modelInfo?.type ?? "ann"} inferenceTime={inferenceTime} isCanvasEmpty={canvasEmpty} />
    </div>
  );
};

export default App;
