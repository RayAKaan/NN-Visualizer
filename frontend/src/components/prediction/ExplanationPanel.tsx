import { ANNExplanation, CNNExplanation, RNNExplanation } from "../../types";

export default function ExplanationPanel({ explanation, confidenceLevel }: { explanation: ANNExplanation | CNNExplanation | RNNExplanation | undefined; confidenceLevel: string }) {
  if (!explanation) return <div className="card">No explanation yet.</div>;
  return <div className="card"><h4>Explanation</h4>{explanation.model_type === "ann" && <div>Top neurons: {explanation.top_neurons.map(n => n.id).join(", ")}<br />Evidence: {explanation.quadrant_evidence}</div>}{explanation.model_type === "cnn" && <div>Active filters: {explanation.active_filters.map(f => `${f.layer}:${f.filter}`).join(", ")}<br />Spatial: {explanation.spatial_evidence}</div>}{explanation.model_type === "rnn" && <div>Timestep importance: {explanation.timestep_importance.join(", ")}<br />Sequence: {explanation.sequential_summary}</div>}<div>Confidence: {confidenceLevel}</div><div>{explanation.uncertainty_notes}</div></div>;
}
