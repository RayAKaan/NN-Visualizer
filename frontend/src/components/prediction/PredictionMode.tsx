import React, { useState, useEffect } from "react";
import { ModelType, PredictionResult, CNNPredictionResult, Edge } from "../../types";
import { usePrediction } from "../../hooks/usePrediction";
import { useNetworkState } from "../../hooks/useNetworkState";
import { useWeights } from "../../hooks/useWeights";
import LeftPanel from "../layout/LeftPanel";
import RightPanel from "../layout/RightPanel";
import OutputBars from "./OutputBars";
import CNNExplanationPanel from "./CNNExplanationPanel";
import NetworkView2D from "./NetworkView2D";
import CNNNetworkView2D from "./CNNNetworkView2D";
import ComparisonView from "./ComparisonView";
import Network3D from "../visualization3d/Network3D";

interface Props {
  apiBase: string;
  view: "2d" | "3d" | "compare";
  activeModel: ModelType;
  canCompare: boolean;
  onCanvasEmptyChange: (empty: boolean) => void;
  onInferenceTimeChange: (time: number) => void;
}

export default function PredictionMode({ apiBase, view, activeModel, onCanvasEmptyChange, onInferenceTimeChange }: Props) {
  const [pixels, setPixels] = useState<number[]>(new Array(784).fill(0));
  const [brushSize, setBrushSize] = useState<1 | 2 | 3>(2);
  const { result, inferenceTime, predict, clear } = usePrediction(apiBase);
  const { state, fetchState } = useNetworkState();
  const weights = useWeights(activeModel);
  const [annResult, setAnnResult] = useState<PredictionResult | null>(null);
  const [cnnResult, setCnnResult] = useState<CNNPredictionResult | null>(null);

  const hasPixels = pixels.some((p) => p > 0);

  useEffect(() => {
    onCanvasEmptyChange(!hasPixels);
  }, [hasPixels, onCanvasEmptyChange]);

  useEffect(() => {
    onInferenceTimeChange(inferenceTime);
  }, [inferenceTime, onInferenceTimeChange]);

  useEffect(() => {
    if (!hasPixels) {
      clear();
      setAnnResult(null);
      setCnnResult(null);
      return;
    }

    if (view === "compare") {
      fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels, model_type: "ann" }),
      }).then((r) => r.json()).then((d) => setAnnResult(d)).catch(() => setAnnResult(null));

      fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels, model_type: "cnn" }),
      }).then((r) => r.json()).then((d) => setCnnResult(d)).catch(() => setCnnResult(null));
    } else {
      predict(pixels, activeModel);
    }

    if (view === "3d") fetchState(pixels, "ann");
  }, [pixels, activeModel, view, hasPixels, clear, predict, fetchState]);

  const handleClear = () => {
    setPixels(new Array(784).fill(0));
    clear();
    setAnnResult(null);
    setCnnResult(null);
  };

  const prediction = result?.prediction ?? -1;
  const probabilities = result?.probabilities ?? new Array(10).fill(0);
  const isANN = activeModel === "ann";
  const hidden1 = isANN && result?.model_type === "ann" ? (result as PredictionResult).layers.hidden1 : [];
  const hidden2 = isANN && result?.model_type === "ann" ? (result as PredictionResult).layers.hidden2 : [];

  const cnnFeatureMaps = !isANN && result?.model_type === "cnn" ? (result as CNNPredictionResult).feature_maps : [];
  const cnnKernels = !isANN && result?.model_type === "cnn" ? (result as CNNPredictionResult).kernels : [];
  const cnnDenseLayers = !isANN && result?.model_type === "cnn" ? (result as CNNPredictionResult).dense_layers : {};

  const edgesH1H2: Edge[] = state?.edges?.hidden1_hidden2 ?? [];
  const edgesH2Out: Edge[] = state?.edges?.hidden2_output ?? [];
  const stateHidden1 = state?.layers?.hidden1 ?? hidden1;
  const stateHidden2 = state?.layers?.hidden2 ?? hidden2;
  const stateOutput = state?.probabilities ?? probabilities;

  const renderCenter = () => {
    if (!hasPixels) return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)" }}>Draw a digit on the canvas to begin</div>;

    if (view === "2d") {
      if (isANN) return <NetworkView2D pixels={pixels} hidden1={hidden1} hidden2={hidden2} probabilities={probabilities} prediction={prediction} weights={weights} />;
      return <CNNNetworkView2D pixels={pixels} featureMaps={cnnFeatureMaps} kernels={cnnKernels} denseLayers={cnnDenseLayers} probabilities={probabilities} prediction={prediction} />;
    }

    if (view === "3d") {
      return <Network3D hidden1={stateHidden1} hidden2={stateHidden2} output={stateOutput} prediction={prediction} edgesH1H2={edgesH1H2} edgesH2Out={edgesH2Out} />;
    }

    return <ComparisonView pixels={pixels} annResult={annResult} cnnResult={cnnResult} />;
  };

  return (
    <>
      <LeftPanel pixels={pixels} onPixelsChange={setPixels} brushSize={brushSize} onBrushSizeChange={setBrushSize} onClear={handleClear} onSampleLoad={(samplePixels) => setPixels([...samplePixels])} activePixelCount={pixels.filter((p) => p > 0).length} />
      <div className="center-stage">{renderCenter()}</div>
      {activeModel === "cnn" ? (
        <aside className="right-stack">
          <OutputBars probabilities={probabilities} winner={prediction} />
          <CNNExplanationPanel explanation={(result as CNNPredictionResult | null)?.explanation ?? null} />
        </aside>
      ) : (
        <RightPanel result={(result as PredictionResult | null) ?? null} />
      )}
    </>
  );
}
