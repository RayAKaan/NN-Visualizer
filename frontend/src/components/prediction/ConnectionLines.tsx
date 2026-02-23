import React, { useMemo } from "react";

interface Props {
  hidden1: number[];
  hidden2: number[];
  probabilities: number[];
  weights: Record<string, { kernel: number[][] }> | null;
  width: number;
  height: number;
}

interface Connection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  strength: number;
}

const ConnectionLines = React.memo(function ConnectionLines({ hidden1, hidden2, probabilities, weights, width, height }: Props) {
  const connections = useMemo(() => {
    if (!weights) return [];

    const lines: Connection[] = [];
    const w_h1h2 = weights.hidden2?.kernel;
    if (w_h1h2 && hidden1.length > 0 && hidden2.length > 0) {
      const candidates: Connection[] = [];
      for (let i = 0; i < Math.min(hidden1.length, w_h1h2.length); i += 1) {
        for (let j = 0; j < Math.min(hidden2.length, w_h1h2[i]?.length || 0); j += 1) {
          const strength = w_h1h2[i][j] * (hidden1[i] || 0);
          if (Math.abs(strength) > 0.01) {
            candidates.push({
              x1: width * 0.28,
              y1: (i / hidden1.length) * height * 0.8 + height * 0.1,
              x2: width * 0.52,
              y2: (j / hidden2.length) * height * 0.8 + height * 0.1,
              strength,
            });
          }
        }
      }
      candidates.sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength));
      lines.push(...candidates.slice(0, 80));
    }

    const w_h2out = weights.output?.kernel;
    if (w_h2out && hidden2.length > 0 && probabilities.length > 0) {
      const candidates: Connection[] = [];
      for (let i = 0; i < Math.min(hidden2.length, w_h2out.length); i += 1) {
        for (let j = 0; j < Math.min(probabilities.length, w_h2out[i]?.length || 0); j += 1) {
          const strength = w_h2out[i][j] * (hidden2[i] || 0);
          if (Math.abs(strength) > 0.01) {
            candidates.push({
              x1: width * 0.55,
              y1: (i / hidden2.length) * height * 0.8 + height * 0.1,
              x2: width * 0.78,
              y2: (j / probabilities.length) * height * 0.6 + height * 0.2,
              strength,
            });
          }
        }
      }
      candidates.sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength));
      lines.push(...candidates.slice(0, 80));
    }

    return lines;
  }, [hidden1, hidden2, probabilities, weights, width, height]);

  return (
    <svg className="connection-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} data-highlight="connections">
      {connections.map((c, i) => {
        const absStr = Math.abs(c.strength);
        const sw = 0.5 + Math.min(absStr * 5, 3.5);
        const opacity = 0.1 + Math.min(absStr * 3, 0.8);
        const color = c.strength > 0 ? `rgba(6,182,212,${opacity})` : `rgba(139,92,246,${opacity})`;
        const cx = (c.x1 + c.x2) / 2;
        const cy = (c.y1 + c.y2) / 2;
        return <path key={i} className="connection-line" d={`M ${c.x1},${c.y1} Q ${cx},${cy} ${c.x2},${c.y2}`} stroke={color} strokeWidth={sw} fill="none" />;
      })}
    </svg>
  );
});

export default ConnectionLines;
