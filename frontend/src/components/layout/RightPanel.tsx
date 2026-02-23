import { AnyPredictionResult, ModelType } from "../../types";
import OutputBars from "../prediction/OutputBars";
import PredictionCard from "../prediction/PredictionCard";
import ExplanationPanel from "../prediction/ExplanationPanel";

type Props = { prediction: number; confidence: number; confidenceLevel: string; probabilities: number[]; explanation: AnyPredictionResult["explanation"] | undefined; loading: boolean; inferenceTime: number | null; hasResult: boolean; pixels: number[]; activeModel: ModelType; onExport: () => void };

export default function RightPanel({ prediction, confidence, confidenceLevel, probabilities, explanation, loading, inferenceTime, hasResult, onExport }: Props) {
  return <div className="right-panel"><PredictionCard prediction={prediction} confidence={confidence} loading={loading} /><button className="btn export block" disabled={!hasResult} onClick={onExport}>Export</button><OutputBars probabilities={probabilities} winner={prediction} /><ExplanationPanel explanation={explanation} confidenceLevel={confidenceLevel} /><div className="muted">Inference: {inferenceTime ? `${inferenceTime.toFixed(1)}ms` : "-"}</div></div>;
}
