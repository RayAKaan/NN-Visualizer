import React, { useMemo } from "react";

interface ConnectionViewProps {
  hidden1: number[];
  hidden2: number[];
  output: number[];
  weightsHidden1Hidden2: number[][] | null;
  weightsHidden2Output: number[][] | null;
  width?: number;
  height?: number;
}

interface NodePosition {
  x: number;
  y: number;
}

interface Connection {
  from: NodePosition;
  to: NodePosition;
  signed: number;
  magnitude: number;
}

/* ---------- layout ---------- */

const buildPositions = (
  count: number,
  x: number,
  height: number
): NodePosition[] => {
  const spacing = height / (count + 1);
  return Array.from({ length: count }, (_, i) => ({
    x,
    y: spacing * (i + 1),
  }));
};

/* ---------- signal extraction ---------- */

const extractConnections = (
  weights: number[][],
  activations: number[],
  fromPos: NodePosition[],
  toPos: NodePosition[],
  limit: number
): Connection[] => {
  const acc: Connection[] = [];

  for (let i = 0; i < weights.length; i++) {
    const a = activations[i] ?? 0;
    if (a < 1e-4) continue;

    for (let j = 0; j < weights[i].length; j++) {
      const w = weights[i][j];
      const signed = w * a;
      const mag = Math.abs(signed);

      if (mag < 0.02) continue;

      acc.push({
        from: fromPos[i],
        to: toPos[j],
        signed,
        magnitude: mag,
      });
    }
  }

  return acc
    .sort((a, b) => b.magnitude - a.magnitude)
    .slice(0, limit);
};

/* ---------- path helper ---------- */

const curvePath = (from: NodePosition, to: NodePosition) => {
  const dx = to.x - from.x;
  const cx1 = from.x + dx * 0.4;
  const cx2 = from.x + dx * 0.6;

  return `M ${from.x} ${from.y}
          C ${cx1} ${from.y},
            ${cx2} ${to.y},
            ${to.x} ${to.y}`;
};

/* ---------- component ---------- */

const ConnectionView: React.FC<ConnectionViewProps> = ({
  hidden1,
  hidden2,
  output,
  weightsHidden1Hidden2,
  weightsHidden2Output,
  width = 720,
  height = 220,
}) => {
  const { h1, h2, out, connections } = useMemo(() => {
    const h1 = buildPositions(hidden1.length, 90, height);
    const h2 = buildPositions(hidden2.length, width / 2, height);
    const out = buildPositions(output.length, width - 80, height);

    const connections: Connection[] = [];

    if (weightsHidden1Hidden2) {
      connections.push(
        ...extractConnections(weightsHidden1Hidden2, hidden1, h1, h2, 160)
      );
    }

    if (weightsHidden2Output) {
      connections.push(
        ...extractConnections(weightsHidden2Output, hidden2, h2, out, 100)
      );
    }

    return { h1, h2, out, connections };
  }, [hidden1, hidden2, output, weightsHidden1Hidden2, weightsHidden2Output, width, height]);

  return (
    <svg className="connection-svg" viewBox={`0 0 ${width} ${height}`}>
      {/* connections */}
      {connections.map((c, i) => {
        const energy = Math.pow(c.magnitude, 0.55); // gamma correction
        const strokeWidth = 0.6 + energy * 3.6;
        const opacity = Math.min(0.9, 0.25 + energy);

        const color =
          c.signed >= 0
            ? `rgba(122, 230, 148, ${opacity})`
            : `rgba(161, 98, 219, ${opacity})`;

        const path = curvePath(c.from, c.to);

        return (
          <g key={`conn-${i}`}>
            {/* soft glow */}
            <path
              d={path}
              stroke={color}
              strokeWidth={strokeWidth + 1.2}
              fill="none"
              opacity={0.15}
            />
            {/* core signal */}
            <path
              d={path}
              stroke={color}
              strokeWidth={strokeWidth}
              fill="none"
              strokeLinecap="round"
            />
          </g>
        );
      })}

      {/* nodes */}
      {h1.map((p, i) => (
        <circle key={`h1-${i}`} cx={p.x} cy={p.y} r={4} fill="#1f2430" />
      ))}
      {h2.map((p, i) => (
        <circle key={`h2-${i}`} cx={p.x} cy={p.y} r={4} fill="#1f2430" />
      ))}
      {out.map((p, i) => (
        <circle key={`out-${i}`} cx={p.x} cy={p.y} r={5} fill="#1f2430" />
      ))}
    </svg>
  );
};

export default ConnectionView;