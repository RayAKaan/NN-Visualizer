import React from "react";

interface Props {
  activations: number[];
  columns: number;
  label: string;
}

const colorFor = (v: number) => {
  const clamped = Math.max(0, Math.min(1, v));
  const r = Math.round(26 + (6 - 26) * clamped + (139 - 6) * clamped * 0.5);
  const g = Math.round(32 + (182 - 32) * clamped * 0.8 + (92 - 182) * clamped * 0.2);
  const b = Math.round(53 + (212 - 53) * clamped * 0.6 + (246 - 212) * clamped * 0.4);
  return `rgb(${r}, ${g}, ${b})`;
};

const NeuronGrid: React.FC<Props> = ({ activations, columns, label }) => {
  return (
    <div className="layer-card">
      <div className="layer-label">{label}</div>
      <div className="neuron-grid" style={{ gridTemplateColumns: `repeat(${columns}, 10px)` }}>
        {activations.map((value, index) => (
          <div
            key={index}
            className="neuron"
            style={{ backgroundColor: colorFor(value) }}
            title={`Neuron #${index} | Activation: ${value.toFixed(3)}`}
          />
        ))}
      </div>
    </div>
  );
};

export default NeuronGrid;
