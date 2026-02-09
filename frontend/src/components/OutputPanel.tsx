import React, { useMemo } from "react";

interface OutputPanelProps {
  probabilities: number[];
}

/* =====================================================
   Helpers

const clamp = (v: number, min = 0, max = 1) =>
  Math.min(max, Math.max(min, v));

/* =====================================================
   Component

const OutputPanel: React.FC<OutputPanelProps> = ({
  probabilities,
}) => {
  const safeProbs = probabilities ?? [];

  const maxIndex = useMemo(() => {
    if (safeProbs.length === 0) return -1;

    let max = 0;
    for (let i = 1; i < safeProbs.length; i++) {
      if (safeProbs[i] > safeProbs[max]) max = i;
    }
    return max;
  }, [safeProbs]);

  return (
    <div className="output-panel">
      {safeProbs.map((value, index) => {
        const v = clamp(value);
        const pct = Math.round(v * 1000) / 10;

        const isActive = index === maxIndex;
        const glow = isActive ? v * 14 : 0;

        return (
          <div
            key={index}
            className={`output-bar ${
              isActive ? "active" : ""
            }`}
          >
            <span className="digit">{index}</span>

            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${pct}%`,
                  opacity: 0.35 + v * 0.65,
                  boxShadow: isActive
                    ? `0 0 ${glow}px rgba(76,201,240,${
                        v * 0.45
                      })`
                    : "none",
                  transition:
                    "width 140ms ease-out, opacity 140ms linear",
                }}
              />
            </div>

            <span className="percent">
              {pct.toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
};

export default React.memo(OutputPanel);
