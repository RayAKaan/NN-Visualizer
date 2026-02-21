import React from "react";

interface Edge { from: number; to: number; strength: number }

const ConnectionLines: React.FC<{ edges: Edge[]; width: number; height: number }> = ({ edges, width, height }) => {
  return (
    <svg className="connection-svg" viewBox={`0 0 ${width} ${height}`}>
      {edges.map((edge, idx) => {
        const x1 = 120;
        const y1 = 20 + (edge.from % 16) * 8;
        const x2 = width - 200;
        const y2 = 20 + (edge.to % 8) * 14;
        const strength = Math.abs(edge.strength);
        return (
          <path
            key={idx}
            d={`M ${x1} ${y1} C ${width * 0.45} ${y1}, ${width * 0.55} ${y2}, ${x2} ${y2}`}
            stroke={edge.strength >= 0 ? "rgba(6,182,212,0.6)" : "rgba(139,92,246,0.6)"}
            strokeWidth={Math.min(4, 0.5 + strength * 3)}
            strokeOpacity={Math.min(0.8, 0.1 + strength * 0.7)}
            fill="none"
          />
        );
      })}
    </svg>
  );
};

export default ConnectionLines;
