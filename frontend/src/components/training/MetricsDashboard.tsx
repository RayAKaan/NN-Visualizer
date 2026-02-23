import React from "react";

type Props = {
  latestLoss?: number;
  latestAccuracy?: number;
  latestValLoss?: number;
  latestValAccuracy?: number;
  latestGradientNorm?: number;
  latestLearningRate?: number;
};

export default function MetricsDashboard({ latestLoss = 0, latestAccuracy = 0, latestValLoss = 0, latestValAccuracy = 0, latestGradientNorm = 0, latestLearningRate = 0 }: Props) {
  const card = (label: string, value: string) => (
    <div className="metrics-card" style={{ padding: 10 }}>
      <div className="control-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );

  return (
    <div className="card">
      <h3>📈 Metrics Dashboard</h3>
      <div style={{ display: "grid", gap: 8 }}>
        {card("Train Loss", latestLoss.toFixed(4))}
        {card("Train Accuracy", `${(latestAccuracy * 100).toFixed(2)}%`)}
        {card("Val Loss", latestValLoss.toFixed(4))}
        {card("Val Accuracy", `${(latestValAccuracy * 100).toFixed(2)}%`)}
        {card("Grad Norm", latestGradientNorm.toFixed(4))}
        {card("Learning Rate", latestLearningRate.toFixed(6))}
      </div>
    </div>
  );
}
