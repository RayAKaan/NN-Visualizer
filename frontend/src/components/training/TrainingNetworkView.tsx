import React from "react";

type Props = {
  gradients: number[];
};

export default function TrainingNetworkView({ gradients }: Props) {
  const max = Math.max(0.0001, ...gradients);
  return (
    <div className="card training-network">
      <h3>🧠 Gradient Flow</h3>
      <div style={{ display: "grid", gap: 6 }}>
        {gradients.slice(-16).map((g, i) => (
          <div key={i} className="metrics-row" style={{ display: "grid", gridTemplateColumns: "80px 1fr 80px", alignItems: "center", gap: 8 }}>
            <span>L{i + 1}</span>
            <div style={{ background: "#0b1220", borderRadius: 999, height: 8 }}><div className="gradient-flow-line" style={{ width: `${(g / max) * 100}%`, background: "#06b6d4", height: "100%" }} /></div>
            <span>{g.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
