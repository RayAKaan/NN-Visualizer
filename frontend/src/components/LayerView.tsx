import React from "react";

interface LayerViewProps {
  title: string;
  activations: number[];
  columns: number;
  showTitle?: boolean;
}

const LayerView: React.FC<LayerViewProps> = ({ title, activations, columns, showTitle = true }) => {
  const normalized = activations.map((value) => Math.min(1, Math.max(0, value)));
  const gridTemplate = `repeat(${columns}, 12px)`;

  return (
    <div className="layer">
      {showTitle && <strong>{title}</strong>}
      <div className="layer-grid" style={{ gridTemplateColumns: gridTemplate }}>
        {normalized.map((value, index) => (
          <div
            key={`${title}-${index}`}
            className="neuron"
            style={{
              background: `rgba(31, 36, 48, ${Math.max(0.12, value)})`,
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default LayerView;
