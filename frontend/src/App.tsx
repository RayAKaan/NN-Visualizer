import { useState, useEffect, useCallback } from "react";
import TopBar from "./components/layout/TopBar";
import StatusBar from "./components/layout/StatusBar";
import PredictionMode from "./components/prediction/PredictionMode";
import TrainingMode from "./components/training/TrainingMode";
import PresentationMode from "./components/presentation/PresentationMode";
import ShortcutHelp from "./components/ui/ShortcutHelp";
import ErrorBanner from "./components/ui/ErrorBanner";
import { useModelInfo } from "./hooks/useModelInfo";
import { useHealthCheck } from "./hooks/useHealthCheck";
import { AppMode, ViewMode } from "./types";

export default function App() {
  const [mode, setMode] = useState<AppMode>("predict");
  const [view, setView] = useState<ViewMode>("2d");
  const { info, available, switchModel } = useModelInfo();
  const { backendOnline, retryCount, check } = useHealthCheck();
  const [presentationActive, setPresentationActive] = useState(false);
  const [shortcutHelpVisible, setShortcutHelpVisible] = useState(false);
  const [clearSignal, setClearSignal] = useState(0);
  const [sampleLoadSignal, setSampleLoadSignal] = useState<number | null>(null);
  const [trainingToggleSignal, setTrainingToggleSignal] = useState(0);

  const handlePresentationAction = useCallback((action: string) => {
    if (action === "switchTo3D") { setView("3d"); setMode("predict"); }
    if (action === "switchToCNN" && available.available.includes("cnn")) switchModel("cnn");
    if (action === "switchToRNN" && available.available.includes("rnn")) switchModel("rnn");
    if (action === "switchToCompare") { setView("compare"); setMode("predict"); }
    if (action === "switchToTraining") setMode("train");
    if (action === "switchToPredict") { setMode("predict"); setView("2d"); }
  }, [available, switchModel]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) || presentationActive) return;
      switch (e.key.toLowerCase()) {
        case "c": setClearSignal((p) => p + 1); break; case "d": setView("2d"); break; case "t": setView("3d"); break;
        case "k": setView("compare"); break; case "m": { const models = available.available; if (models.length >= 2) { const idx = models.indexOf(available.active); switchModel(models[(idx + 1) % models.length]); } break; }
        case "p": setPresentationActive(true); break; case " ": if (mode === "train") { e.preventDefault(); setTrainingToggleSignal((p) => p + 1); } break;
        case "?": e.preventDefault(); setShortcutHelpVisible(true); break; case "escape": setShortcutHelpVisible(false); setPresentationActive(false); break;
      }
      if (e.key >= "0" && e.key <= "9" && mode === "predict") setSampleLoadSignal(parseInt(e.key));
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [presentationActive, mode, available, switchModel]);

  useEffect(() => { if (sampleLoadSignal !== null) { const t = setTimeout(() => setSampleLoadSignal(null), 100); return () => clearTimeout(t); } }, [sampleLoadSignal]);

  return <div className="app-layout"><TopBar mode={mode} onModeChange={setMode} view={view} onViewChange={setView} activeModel={available.active} availableModels={available.available} onModelSwitch={switchModel} modelInfo={info} onPresentationStart={() => setPresentationActive(true)} backendOnline={backendOnline} />
    {!backendOnline && <div style={{ padding: "0 16px" }}><ErrorBanner message="Backend not responding. Run: uvicorn app:app --port 8000" type="error" onRetry={check} retryCount={retryCount} /></div>}
    <div className="main-content">{mode === "predict" ? <PredictionMode view={view} activeModel={available.active} availableModels={available.available} clearSignal={clearSignal} sampleLoadSignal={sampleLoadSignal} /> : <TrainingMode trainingToggleSignal={trainingToggleSignal} availableModels={available.available} />}</div>
    <StatusBar activeModel={available.active} backendOnline={backendOnline} />
    <PresentationMode active={presentationActive} onExit={() => setPresentationActive(false)} onAction={handlePresentationAction} />
    <ShortcutHelp visible={shortcutHelpVisible} onClose={() => setShortcutHelpVisible(false)} /></div>;
}
