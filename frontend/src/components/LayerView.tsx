import React, { useMemo } from "react";

interface LayerViewProps {
  title: string;
  activations: number[];
  columns: number;
  showTitle?: boolean;
}

/* ---------- helpers ---------- */

const normalizeLayer = (values: number[]) => {
  const max = Math.max(...values, 1e-6);
  return values.map((v) => v / max);
};

const gammaCorrect = (v: number, gamma = 0.6) => Math.pow(v, gamma);

/* ---------- component ---------- */

const LayerView: React.FC<LayerViewProps> = ({
  title,
  activations,
  columns,
  showTitle = true,
}) => {
  const normalized = useMemo(() => {
    const norm = normalizeLayer(activations);
    return norm.map((v) => gammaCorrect(Math.min(1, Math.max(0, v))));
  }, [activations]);

  const gridTemplate = `repeat(${columns}, 14px)`; // slightly more breathable

  return (
    <div className="layer">
      {showTitle && <strong>{title}</strong>}

      <div className="layer-grid" style={{ gridTemplateColumns: gridTemplate }}>
        {normalized.map((value, index) => {
          const opacity = 0.12 + value * 0.88;
          const glow = value > 0.75 ? value * 6 : 0;

          return (
            <div
              key={`${title}-${index}`}
              className="neuron"
              style={{
                background: `rgba(31, 36, 48, ${opacity})`,
                boxShadow:
                  glow > 0
                    ? `0 0 ${glow}px rgba(31, 36, 48, ${value * 0.35})`
                    : "none",
                borderColor: `rgba(31, 36, 48, ${0.25 + value * 0.55})`,
              }}
            />
          );
        })}
      </div>
    </div>
  );
};

export default LayerView;