import { ModelType } from "../../types";

export default function StatusBar({ activeModel, backendOnline, inferenceTime }: { activeModel: ModelType; backendOnline: boolean; inferenceTime?: number | null }) {
  return <footer className="status-bar"><span>{backendOnline ? "🟢 Connected" : "🔴 Disconnected"}</span><span>Model: {activeModel.toUpperCase()}</span><span>Inference: {inferenceTime ? `${inferenceTime.toFixed(1)}ms` : "Draw a digit to begin"}</span></footer>;
}
