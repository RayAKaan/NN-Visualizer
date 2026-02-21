import React, { useEffect, useMemo, useState } from "react";
import LeftPanel from "../layout/LeftPanel";
import RightPanel from "../layout/RightPanel";
import NetworkView2D from "./NetworkView2D";
import Network3D from "../visualization3d/Network3D";
import { usePrediction } from "../../hooks/usePrediction";

interface Props {
  apiBase: string;
  view: "2d" | "3d";
  onCanvasEmptyChange: (empty: boolean) => void;
  onInferenceTimeChange: (time: number) => void;
}

const PredictionMode: React.FC<Props> = ({ apiBase, view, onCanvasEmptyChange, onInferenceTimeChange }) => {
  const [pixels, setPixels] = useState<number[]>(Array.from({ length: 784 }, () => 0));
  const [brushSize, setBrushSize] = useState<1 | 2 | 3>(2);
  const { result, state3D, inferenceTime, predict } = usePrediction(apiBase);

  useEffect(() => {
    predict(pixels);
    onCanvasEmptyChange(pixels.every((v) => v <= 0.01));
  }, [pixels, predict, onCanvasEmptyChange]);

  useEffect(() => {
    onInferenceTimeChange(inferenceTime);
  }, [inferenceTime, onInferenceTimeChange]);

  const center = useMemo(() => {
    if (view === "3d") {
      return <Network3D state={state3D} />;
    }
    return <NetworkView2D state={state3D} />;
  }, [view, state3D]);

  return (
    <div className="main-content">
      <LeftPanel
        pixels={pixels}
        onPixelsChange={setPixels}
        brushSize={brushSize}
        onBrushSizeChange={setBrushSize}
      />
      <section className="center-stage">{center}</section>
      <RightPanel result={result} />
    </div>
  );
};

export default PredictionMode;
