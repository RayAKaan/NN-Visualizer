import React, { useMemo } from "react";

/* =====================================================
   Types
===================================================== */

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

/* =====================================================
   Layout helpers
===================================================== */

const buildPositions = (
  count: number,
  x: number,
  height: number
): NodePosition[] => {
  if (count <= 0) return [];
  const spacing = height / (count + 1);
  return Array.from({ length: count }, (_, i) => ({
    x,
    y: spacing * (i + 1),
  }));
};

/* =====================================================
   Connection extraction (READABLE + SAFE)
===================================================== */

const extractConnections = (
  weights: unknown,
  activations: number[],
  fromPos: NodePosition[],
  toPos: NodePosition[],
  perNeuronLimit: number,
  globalLimit: number
): Connection[] => {
  if (!Array.isArray(weights)) return [];

  const all: Connection[] = [];

  for (let i = 0; i < weights.length; i++) {
    const row = weights[i];

    // 🔒 HARD GUARD
    if (!Array.isArray(row)) continue;

    const a = activations[i] ?? 0;
    if (Math.abs(a) < 0.06) continue;

    const local: Connection[] = [];

    for (let j = 0; j < row.length; j++) {
      const w = row[j];
      if (typeof w !== "number") continue;

      const signed = w * a;
      const magnitude = Math.abs(signed);
      if (magnitude < 0.05) continue;

      local.push({
        from: fromPos[i],
        to: toPos[j],
        signed,
        magnitude,
      });
    }

    local.sort((a, b) => b.magnitude - a.magnitude);
    all.push(...local.slice(0, perNeuronLimit));
  }

  all.sort((a, b) => b.magnitude - a.magnitude);
  return all.slice(0, globalLimit);
};

/* =====================================================
   SVG path cache
===================================================== */

const pathCache = new Map<string, string>();

const curvePath = (from: NodePosition, to: NodePosition): string => {
  const key = `${from.x}|${from.y}|${to.x}|${to.y}`;
  const cached = pathCache.get(key);
  if (cached) return cached;

  const dx = to.x - from.x;
  const cx1 = from.x + dx * 0.45;
  const cx2 = from.x + dx * 0.55;

  const path = `M ${from.x} ${from.y}
                C ${cx1} ${from.y},
                  ${cx2} ${to.y},
                  ${to.x} ${to.y}`;

  pathCache.set(key, path);
  return path;
};

/* =====================================================
   Component
===================================================== */

const ConnectionView: React.FC<ConnectionViewProps> = ({
  hidden1,
  hidden2,
  output,
  weightsHidden1Hidden2,
  weightsHidden2Output,
  width = 900,
  height = 320,
}) => {
  const { h1, h2, out, connections } = useMemo(() => {
    const h1 = buildPositions(hidden1.length, 90, height);
    const h2 = buildPositions(hidden2.length, width * 0.5, height);
    const out = buildPositions(output.length, width - 80, height);

    const connections: Connection[] = [];

    if (weightsHidden1Hidden2) {
      connections.push(
        ...extractConnections(
          weightsHidden1Hidden2,
          hidden1,
          h1,
          h2,
          3,    // per hidden1 neuron
          140   // global cap
        )
      );
    }

    if (weightsHidden2Output) {
      connections.push(
        ...extractConnections(
          weightsHidden2Output,
          hidden2,
          h2,
          out,
          4,    // per hidden2 neuron
          70
        )
      );
    }

    return { h1, h2, out, connections };
  }, [
    hidden1,
    hidden2,
    output,
    weightsHidden1Hidden2,
    weightsHidden2Output,
    width,
    height,
  ]);

  return (
    <svg
      className="connection-svg"
      viewBox={`0 0 ${width} ${height}`}
      shapeRendering="geometricPrecision"
    >
      {/* Connections */}
      {connections.map((c, i) => {
        const energy = Math.pow(c.magnitude, 0.6);
        const strokeWidth = 0.8 + energy * 2.6;
        const opacity = Math.min(0.8, energy * 0.9);

        const color =
          c.signed >= 0
            ? `rgba(76,201,240,${opacity})`
            : `rgba(199,125,255,${opacity})`;

        const path = curvePath(c.from, c.to);

        return (
          <g key={`conn-${i}`}>
            <path
              d={path}
              stroke={color}
              strokeWidth={strokeWidth + 1.6}
              fill="none"
              opacity={0.18}
            />
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

      {/* Nodes */}
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

export default React.memo(ConnectionView);