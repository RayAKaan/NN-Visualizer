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
  weight: number;
  intensity: number;
}

const buildPositions = (count: number, x: number, height: number) => {
  const spacing = height / (count + 1);
  return Array.from({ length: count }, (_, index) => ({
    x,
    y: spacing * (index + 1),
  }));
};

const topConnections = (
  weights: number[][],
  sourceActivations: number[],
  sourcePositions: NodePosition[],
  targetPositions: NodePosition[],
  maxConnections: number
): Connection[] => {
  const flat: Connection[] = [];
  for (let i = 0; i < weights.length; i += 1) {
    for (let j = 0; j < weights[i].length; j += 1) {
      const weight = weights[i][j];
      const activation = sourceActivations[i] ?? 0;
      const intensity = weight * activation;
      if (Math.abs(intensity) < 0.05) continue;
      flat.push({
        from: sourcePositions[i],
        to: targetPositions[j],
        weight,
        intensity,
      });
    }
  }
  return flat
    .sort((a, b) => Math.abs(b.intensity) - Math.abs(a.intensity))
    .slice(0, maxConnections);
};

const ConnectionView: React.FC<ConnectionViewProps> = ({
  hidden1,
  hidden2,
  output,
  weightsHidden1Hidden2,
  weightsHidden2Output,
  width = 700,
  height = 360,
}) => {
  const { hidden1Positions, hidden2Positions, outputPositions, connections } = useMemo(() => {
    const hidden1Positions = buildPositions(hidden1.length, 100, height);
    const hidden2Positions = buildPositions(hidden2.length, width / 2, height);
    const outputPositions = buildPositions(output.length, width - 80, height);

    const connections: Connection[] = [];
    if (weightsHidden1Hidden2) {
      connections.push(
        ...topConnections(
          weightsHidden1Hidden2,
          hidden1,
          hidden1Positions,
          hidden2Positions,
          120
        )
      );
    }
    if (weightsHidden2Output) {
      connections.push(
        ...topConnections(
          weightsHidden2Output,
          hidden2,
          hidden2Positions,
          outputPositions,
          80
        )
      );
    }

    return { hidden1Positions, hidden2Positions, outputPositions, connections };
  }, [hidden1, hidden2, output, weightsHidden1Hidden2, weightsHidden2Output, width, height]);

  return (
    <svg className="connection-svg" viewBox={`0 0 ${width} ${height}`}>
      {connections.map((connection, index) => {
        const color =
          connection.intensity >= 0 ? "rgba(122, 230, 148, 0.65)" : "rgba(161, 98, 219, 0.65)";
        const thickness = Math.min(3.5, Math.max(0.6, Math.abs(connection.intensity) * 3.5));
        return (
          <line
            key={`conn-${index}`}
            x1={connection.from.x}
            y1={connection.from.y}
            x2={connection.to.x}
            y2={connection.to.y}
            stroke={color}
            strokeWidth={thickness}
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
