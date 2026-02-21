import React, { useEffect, useMemo, useState } from "react";

interface ConnectionViewProps {
  hidden1: number[];
  hidden2: number[];
  output: number[];
  weightsHidden1Hidden2: number[][] | null;
  weightsHidden2Output: number[][] | null;
  gradientsHidden1Hidden2?: number[][] | null;
  gradientsHidden2Output?: number[][] | null;
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
  intensity: number;
  gradient: number;
}

const buildPositions = (count: number, x: number, height: number) => {
  const spacing = height / (count + 1);
  return Array.from({ length: count }, (_, index) => ({ x, y: spacing * (index + 1) }));
};

const topConnections = (
  weights: number[][],
  gradients: number[][] | null,
  sourceActivations: number[],
  sourcePositions: NodePosition[],
  targetPositions: NodePosition[],
  maxConnections: number
): Connection[] => {
  const flat: Connection[] = [];
  for (let i = 0; i < weights.length; i += 1) {
    for (let j = 0; j < weights[i].length; j += 1) {
      const intensity = (weights[i][j] ?? 0) * (sourceActivations[i] ?? 0);
      const gradient = gradients?.[i]?.[j] ?? 0;
      if (Math.abs(intensity) < 0.01 && Math.abs(gradient) < 0.01) continue;
      flat.push({ from: sourcePositions[i], to: targetPositions[j], intensity, gradient });
    }
  }
  return flat.sort((a, b) => Math.abs(b.intensity) - Math.abs(a.intensity)).slice(0, maxConnections);
};

const ConnectionView: React.FC<ConnectionViewProps> = ({
  hidden1,
  hidden2,
  output,
  weightsHidden1Hidden2,
  weightsHidden2Output,
  gradientsHidden1Hidden2 = null,
  gradientsHidden2Output = null,
  width = 700,
  height = 360,
}) => {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let raf = 0;
    const animate = () => {
      setTick((t) => t + 1);
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, []);

  const { hidden1Positions, hidden2Positions, outputPositions, connections } = useMemo(() => {
    const hidden1Positions = buildPositions(hidden1.length, 100, height);
    const hidden2Positions = buildPositions(hidden2.length, width / 2, height);
    const outputPositions = buildPositions(output.length, width - 80, height);

    const connections: Connection[] = [];
    if (weightsHidden1Hidden2) {
      connections.push(
        ...topConnections(
          weightsHidden1Hidden2,
          gradientsHidden1Hidden2,
          hidden1,
          hidden1Positions,
          hidden2Positions,
          160
        )
      );
    }
    if (weightsHidden2Output) {
      connections.push(
        ...topConnections(
          weightsHidden2Output,
          gradientsHidden2Output,
          hidden2,
          hidden2Positions,
          outputPositions,
          100
        )
      );
    }

    return { hidden1Positions, hidden2Positions, outputPositions, connections };
  }, [
    hidden1,
    hidden2,
    output,
    weightsHidden1Hidden2,
    weightsHidden2Output,
    gradientsHidden1Hidden2,
    gradientsHidden2Output,
    width,
    height,
  ]);

  const pulse = 0.5 + 0.5 * Math.sin(tick * 0.08);

  return (
    <svg className="connection-svg" viewBox={`0 0 ${width} ${height}`}>
      {connections.map((connection, index) => {
        const gradMag = Math.abs(connection.gradient);
        const color = connection.gradient >= 0 ? "rgba(59,130,246,0.7)" : "rgba(239,68,68,0.7)";
        const thickness = Math.max(0.6, Math.min(4.5, gradMag * 2.5 + Math.abs(connection.intensity) * 2));
        const opacity = Math.max(0.15, Math.min(0.95, 0.2 + gradMag * 0.8 * pulse));
        return (
          <line
            key={`conn-${index}`}
            x1={connection.from.x}
            y1={connection.from.y}
            x2={connection.to.x}
            y2={connection.to.y}
            stroke={color}
            strokeWidth={thickness}
            strokeOpacity={opacity}
          />
        );
      })}
      {hidden1Positions.map((pos, index) => (
        <circle key={`h1-${index}`} cx={pos.x} cy={pos.y} r={4} fill="#1f2430" />
      ))}
      {hidden2Positions.map((pos, index) => (
        <circle key={`h2-${index}`} cx={pos.x} cy={pos.y} r={4} fill="#1f2430" />
      ))}
      {outputPositions.map((pos, index) => (
        <circle key={`out-${index}`} cx={pos.x} cy={pos.y} r={5} fill="#1f2430" />
      ))}
    </svg>
  );
};

export default ConnectionView;
