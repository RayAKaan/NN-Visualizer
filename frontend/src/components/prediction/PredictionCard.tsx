export default function PredictionCard({ prediction, confidence, loading }: { prediction: number; confidence: number; loading: boolean }) {
  return <div className="card prediction-card" data-highlight="prediction"><div className="prediction-digit">{loading ? "..." : prediction}</div><div className="confidence-ring">{(confidence * 100).toFixed(1)}%</div></div>;
}
