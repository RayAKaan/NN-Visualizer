import React, { useEffect, useMemo, useState } from "react";
import LeftPanel from "../layout/LeftPanel";
import RightPanel from "../layout/RightPanel";
import NetworkView2D from "./NetworkView2D";
import Network3D from "../visualization3d/Network3D";
import { usePrediction } from "../../hooks/usePrediction";
import { CNNPredictionResult, ModelType, PredictionResult } from "../../types";
import CNNNetworkView2D from "./CNNNetworkView2D";
import CNNExplanationPanel from "./CNNExplanationPanel";
import OutputBars from "./OutputBars";
import ComparisonView from "./ComparisonView";

interface Props {
  apiBase: string;
  view: "2d" | "3d" | "compare";
  activeModel: ModelType;
  canCompare: boolean;
  onCanvasEmptyChange: (empty: boolean) => void;
  onInferenceTimeChange: (time: number) => void;
}

const PredictionMode: React.FC<Props> = ({
  apiBase,
  view,
  activeModel,
  canCompare,
  onCanvasEmptyChange,
  onInferenceTimeChange,
}) => {
  const [pixels, setPixels] = useState<number[]>(Array.from({ length: 784 }, () => 0));
  const [brushSize, setBrushSize] = useState<1 | 2 | 3>(2);
  const { result, state3D, inferenceTime, predict, clear } = usePrediction(apiBase);
  const [annResult, setAnnResult] = useState<PredictionResult | null>(null);
  const [cnnResult, setCnnResult] = useState<CNNPredictionResult | null>(null);

  useEffect(() => {
    clear();
    predict(pixels, activeModel);
  }, [activeModel]);

  useEffect(() => {
    onCanvasEmptyChange(pixels.every((v) => v <= 0.01));
    if (view === "compare" && canCompare) {
      fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels, model_type: "ann" }),
      })
        .then((r) => r.json())
        .then((r) => setAnnResult(r as PredictionResult))
        .catch(() => setAnnResult(null));
      fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels, model_type: "cnn" }),
      })
        .then((r) => r.json())
        .then((r) => setCnnResult(r as CNNPredictionResult))
        .catch(() => setCnnResult(null));
      return;
    }
    predict(pixels, activeModel);
  }, [pixels, predict, activeModel, view, canCompare, apiBase, onCanvasEmptyChange]);

  useEffect(() => {
    onInferenceTimeChange(inferenceTime);
  }, [inferenceTime, onInferenceTimeChange]);

  const center = useMemo(() => {
    if (view === "compare") {
      return <ComparisonView annResult={annResult} cnnResult={cnnResult} />;
    }
    if (view === "3d") {
      return <Network3D state={state3D} />;
    }
    if (activeModel === "cnn") {
      return <CNNNetworkView2D result={result as CNNPredictionResult | null} pixels={pixels} />;
    }
    return <NetworkView2D state={state3D} />;
  }, [view, activeModel, result, state3D, pixels, annResult, cnnResult]);

  const right = useMemo(() => {
    if (activeModel === "cnn") {
      const cnn = result as CNNPredictionResult | null;
      return (
        <aside className="right-stack">
          <OutputBars probabilities={cnn?.probabilities ?? Array.from({ length: 10 }, () => 0)} winner={cnn?.prediction ?? -1} />
          <CNNExplanationPanel explanation={cnn?.explanation ?? null} />
        </aside>
      );
    }
    return <RightPanel result={(result as PredictionResult | null) ?? null} />;
  }, [activeModel, result]);

  return (
    <div className="main-content">
      <LeftPanel
        pixels={pixels}
        onPixelsChange={setPixels}
        brushSize={brushSize}
        onBrushSizeChange={setBrushSize}
      />
      <section className="center-stage">{center}</section>
      {right}
    </div>
  );
};

export default PredictionMode;
