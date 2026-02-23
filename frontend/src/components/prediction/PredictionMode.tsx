import { useEffect, useMemo, useState } from "react";
import LeftPanel from "../layout/LeftPanel";
import RightPanel from "../layout/RightPanel";
import { usePrediction } from "../../hooks/usePrediction";
import { useNetworkState } from "../../hooks/useNetworkState";
import { useWeights } from "../../hooks/useWeights";
import { loadSampleDigits } from "../../utils/sampleDigits";
import { exportPrediction } from "../../utils/exportUtils";
import { ModelType, ViewMode } from "../../types";
import NetworkView2D from "../visualization/NetworkView2D";
import CNNNetworkView2D from "../visualization/CNNNetworkView2D";
import RNNNetworkView2D from "../visualization/RNNNetworkView2D";
import ComparisonView from "../visualization/ComparisonView";
import Network3D from "../visualization3d/Network3D";

export default function PredictionMode({ view, activeModel, availableModels, clearSignal, sampleLoadSignal }: { view: ViewMode; activeModel: ModelType; availableModels: ModelType[]; clearSignal: number; sampleLoadSignal: number | null }) {
  const [pixels, setPixels] = useState<number[]>(Array(784).fill(0));
  const [brushSize, setBrushSize] = useState(2);
  const { result, loading, inferenceTime, predict } = usePrediction();
  const { state, fetchState } = useNetworkState();
  const { weights } = useWeights(activeModel);

  useEffect(() => { predict(pixels, activeModel); fetchState(pixels, activeModel); }, [pixels, activeModel]);
  useEffect(() => { if (clearSignal) setPixels(Array(784).fill(0)); }, [clearSignal]);
  useEffect(() => { if (sampleLoadSignal !== null) loadSampleDigits().then(s => setPixels(s[sampleLoadSignal] || Array(784).fill(0))); }, [sampleLoadSignal]);

  const comparisonResults = useMemo(() => (view === "compare" ? availableModels : []).map((m) => ({ model: m, prediction: result?.model_type === m ? result.prediction : "-" })), [view, availableModels, result]);

  return <>
    <LeftPanel pixels={pixels} onPixelsChange={setPixels} brushSize={brushSize} onBrushSizeChange={setBrushSize} onClear={() => setPixels(Array(784).fill(0))} onSampleLoad={(d) => loadSampleDigits().then(s => setPixels(s[d] || Array(784).fill(0)))} activePixelCount={pixels.filter(Boolean).length} />
    <div className="center-stage">{view === "2d" && activeModel === "ann" && <NetworkView2D state={state} weights={weights} />} {view === "2d" && activeModel === "cnn" && <CNNNetworkView2D result={result} />} {view === "2d" && activeModel === "rnn" && <RNNNetworkView2D result={result} />} {view === "3d" && (activeModel === "ann" ? <Network3D state={state} /> : <div className="stub-placeholder">3D only for ANN in Phase 1</div>)} {view === "compare" && <ComparisonView results={comparisonResults} />}</div>
    <RightPanel prediction={result?.prediction ?? 0} confidence={result?.confidence ?? 0} confidenceLevel={result?.confidence && result.confidence > 0.85 ? "high" : "medium"} probabilities={result?.probabilities ?? Array(10).fill(0)} explanation={result?.explanation} loading={loading} inferenceTime={inferenceTime} hasResult={!!result} pixels={pixels} activeModel={activeModel} onExport={() => exportPrediction(pixels, result?.prediction ?? 0, result?.confidence ?? 0, result?.probabilities ?? Array(10).fill(0))} />
  </>;
}
