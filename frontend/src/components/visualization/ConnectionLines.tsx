import React, { useEffect, useMemo, useState } from "react";

interface Connection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  strength: number;
}

interface Props {
  hidden1: number[];
  hidden2: number[];
  hidden3: number[];
  probabilities: number[];
  weights: Record<string, { kernel: number[][] } | unknown> | null;
  width: number;
  height: number;
}

const TOP_K = 72;

function normalizeKernel(v: unknown): number[][] | undefined {
  if (Array.isArray(v)) return v as number[][];
  if (v && typeof v === "object" && Array.isArray((v as { kernel?: unknown }).kernel)) {
    return (v as { kernel: number[][] }).kernel;
  }
  return undefined;
}

function buildPairConnections(
  sourceActs: number[],
  targetSize: number,
  weightMatrix: number[][] | undefined,
  x1: number,
  x2: number,
  height: number
): Connection[] {
  if (!weightMatrix || sourceActs.length === 0) return [];
  const candidates: Connection[] = [];
  const srcN = sourceActs.length;
  const tgtN = targetSize;

  for (let i = 0; i < Math.min(srcN, weightMatrix.length); i += 1) {
    const act = sourceActs[i] || 0;
    if (Math.abs(act) < 0.001) continue;
    const row = weightMatrix[i];
    if (!row) continue;
    for (let j = 0; j < Math.min(tgtN, row.length); j += 1) {
      const s = row[j] * act;
      if (Math.abs(s) < 0.005) continue;
      candidates.push({
        x1,
        y1: (i / srcN) * height * 0.8 + height * 0.1,
        x2,
        y2: (j / tgtN) * height * 0.8 + height * 0.1,
        strength: s,
      });
    }
  }

  candidates.sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength));
  return candidates.slice(0, TOP_K);
}

const ConnectionLines = React.memo(function ConnectionLines({ hidden1, hidden2, hidden3, probabilities, weights, width, height }: Props) {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    let raf = 0;
    const animate = () => {
      setPhase((p) => (p + 1.4) % 1000);
      raf = window.requestAnimationFrame(animate);
    };
    raf = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(raf);
  }, []);

  const connections = useMemo(() => {
    if (!weights || width === 0 || height === 0) return [];
    const h2 = normalizeKernel(weights.hidden2);
    const h3 = normalizeKernel(weights.hidden3);
    const out = normalizeKernel(weights.output);

    return [
      ...buildPairConnections(hidden1, hidden2.length, h2, width * 0.26, width * 0.44, height),
      ...buildPairConnections(hidden2, hidden3.length, h3, width * 0.44, width * 0.62, height),
      ...buildPairConnections(hidden3, probabilities.length, out, width * 0.62, width * 0.8, height),
    ];
  }, [hidden1, hidden2, hidden3, probabilities, weights, width, height]);

  const flowNodes = useMemo(() => {
    return connections.slice(0, 34).map((c, i) => {
      const t = ((phase / 1000) + ((i * 37) % 100) / 100) % 1;
      const x = c.x1 + (c.x2 - c.x1) * t;
      const y = c.y1 + (c.y2 - c.y1) * t;
      const abs = Math.abs(c.strength);
      const r = 1 + Math.min(abs * 5, 2.8);
      const color = c.strength > 0 ? "rgba(6,182,212,0.9)" : "rgba(139,92,246,0.9)";
      return { x, y, r, color };
    });
  }, [connections, phase]);

  return (
    <svg className="connection-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} data-highlight="connections" style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 1 }}>
      {connections.map((c, i) => {
        const abs = Math.abs(c.strength);
        const sw = 0.5 + Math.min(abs * 5, 3.2);
        const op = 0.08 + Math.min(abs * 3, 0.75);
        const color = c.strength > 0 ? `rgba(6,182,212,${op})` : `rgba(139,92,246,${op})`;
        const cx = (c.x1 + c.x2) / 2;
        const cy = (c.y1 + c.y2) / 2;
        const dash = 5 + Math.min(abs * 10, 12);
        return (
          <path
            key={i}
            className="connection-line"
            d={`M ${c.x1},${c.y1} Q ${cx},${cy} ${c.x2},${c.y2}`}
            stroke={color}
            strokeWidth={sw}
            fill="none"
            strokeDasharray={`${dash} ${dash + 12}`}
            strokeDashoffset={-phase * 0.35}
          />
        );
      })}

      {flowNodes.map((n, i) => (
        <circle key={`node-${i}`} cx={n.x} cy={n.y} r={n.r} fill={n.color} />
      ))}
    </svg>
  );
});

export default ConnectionLines;
