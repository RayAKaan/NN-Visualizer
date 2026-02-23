import React from "react";

function NeuronGrid({ values, highlightId }: { values: number[]; highlightId?: string }) {
  return <div className="neuron-grid" data-highlight={highlightId}>{values.map((v, i) => <div key={i} className="neuron" title={`${i}: ${v.toFixed(3)}`} style={{ background: `rgba(6,182,212,${Math.max(0.12, Math.min(1, v))})`, border: v > 0.8 ? "1px solid #a78bfa" : "1px solid transparent" }} />)}</div>;
}

export default React.memo(NeuronGrid);
