import React, { useMemo } from "react";

interface LayerViewProps {
  title: string;
  activations: number[];
  columns: number;
  showTitle?: boolean;
}

/* =====================================================
   Helpers (robust + perceptual)
===================================================== */

const normalizeLayer = (values: number[]) => {
  if (!values || values.length === 0) return [];
  const max = Math.max(...values, 1e-6);
  return values.map((v) => v / max);
};

// Perceptual gamma → smoother mid-range contrast
const gammaCorrect = (v: number, gamma = 0.65) =>
  Math.pow(Math.min(Math.max(v, 0), 1), gamma);

/* =====================================================
   Component
===================================================== */

const LayerView: React.FC<LayerViewProps> = ({
  title,
  activations,
  columns,
  showTitle = true,
}) => {
  const values = useMemo(() => {
    const norm = normalizeLayer(activations);
    return norm.map((v) => gammaCorrect(v));
  }, [activations]);

  const gridTemplate = useMemo(
    () => `repeat(${columns}, 14px)`,
    [columns]
  );

  return (
    <div className="layer">
      {showTitle && (
        <strong
          style={{
            fontSize: 13,
            opacity: 0.85,
            marginBottom: 4,
          }}
        >
          {title}
        </strong>
      )}

      <div
        className="layer-grid"
        style={{
          gridTemplateColumns: gridTemplate,
        }}
      >
        {values.map((v, i) => {
          const intensity = 0.15 + v * 0.85;
          const glow = v > 0.78 ? (v - 0.78) * 18 : 0;

          return (
            <div
              key={i}
              className="neuron"
              style={{
                background: `rgba(31,36,48,${intensity})`,
                borderColor: `rgba(31,36,48,${
                  0.3 + v * 0.5
                })`,
                boxShadow:
                  glow > 0
                    ? `0 0 ${glow}px rgba(76,201,240,${
                        v * 0.35
                      })`
                    : "none",
                transition:
                  "background 120ms linear, box-shadow 120ms linear",
              }}
            />
          );
        })}
      </div>
    </div>
  );
};

export default React.memo(LayerView);