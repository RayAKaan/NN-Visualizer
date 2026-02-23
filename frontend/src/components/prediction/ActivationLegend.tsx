import React from "react";

export default function ActivationLegend() {
  return (
    <div className="activation-legend">
      <span>0.0</span>
      <div className="legend-gradient" />
      <span>1.0</span>
      <span style={{ marginLeft: 8, color: "var(--text-muted)" }}>Neuron Activation</span>
    </div>
  );
}
