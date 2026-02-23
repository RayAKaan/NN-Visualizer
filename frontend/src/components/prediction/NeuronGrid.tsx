import React from "react";

interface Props {
  activations: number[];
  columns: number;
  label: string;
  highlightId?: string;
}

function activationColor(a: number): string {
  const c = Math.max(0, Math.min(1, a));
  if (c <= 0.5) {
    const t = c / 0.5;
    return `rgb(${Math.round(10 + t * 129)},${Math.round(14 + t * 78)},${Math.round(23 + t * 223)})`;
  }
  const t = (c - 0.5) / 0.5;
  return `rgb(${Math.round(139 + t * (6 - 139))},${Math.round(92 + t * 90)},${Math.round(246 + t * (212 - 246))})`;
}

const NeuronGrid = React.memo(function NeuronGrid({ activations, columns, label, highlightId }: Props) {
  return (
    <div className="layer-block" data-highlight={highlightId}>
      <div className="neuron-grid" style={{ width: columns * 10 }}>
        {activations.map((a, i) => (
          <div
            key={i}
            className="neuron"
            title={`Neuron #${i} | Act: ${(a ?? 0).toFixed(3)}`}
            style={{
              backgroundColor: activationColor(a ?? 0),
              border: (a ?? 0) > 0.7 ? "1px solid rgba(6,182,212,0.5)" : "1px solid transparent",
            }}
          />
        ))}
      </div>
      <span className="layer-label">{label}</span>
    </div>
  );
});

export default NeuronGrid;
