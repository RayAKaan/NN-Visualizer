import React, { useMemo } from "react";

interface LossLandscape3DProps {
  lossHistory: number[];
  weightHistory: number[];
}

const randomProjection = [
  [0.73, -0.21, 0.49],
  [-0.33, 0.88, 0.12],
];

const LossLandscape3D: React.FC<LossLandscape3DProps> = ({ lossHistory, weightHistory }) => {
  const trajectory = useMemo(() => {
    const count = Math.min(lossHistory.length, weightHistory.length);
    return Array.from({ length: count }, (_, i) => {
      const w = weightHistory[i] ?? 0;
      const l = lossHistory[i] ?? 0;
      const x = w * randomProjection[0][0] + l * randomProjection[1][0];
      const y = w * randomProjection[0][1] + l * randomProjection[1][1];
      const z = l * randomProjection[0][2] + w * randomProjection[1][2];
      return { x, y, z };
    });
  }, [lossHistory, weightHistory]);

  return (
    <div className="loss-landscape">
      <strong>Loss landscape (projected)</strong>
      <svg viewBox="0 0 320 180" className="loss-landscape-svg">
        {trajectory.slice(1).map((point, idx) => {
          const prev = trajectory[idx];
          return (
            <line
              key={`traj-${idx}`}
              x1={160 + prev.x * 80}
              y1={120 - prev.y * 60}
              x2={160 + point.x * 80}
              y2={120 - point.y * 60}
              stroke="rgba(14,165,233,0.65)"
              strokeWidth={2}
            />
          );
        })}
        {trajectory.length > 0 && (
          <circle
            cx={160 + trajectory[trajectory.length - 1].x * 80}
            cy={120 - trajectory[trajectory.length - 1].y * 60}
            r={5}
            fill="#ef4444"
          />
        )}
      </svg>
    </div>
  );
};

export default LossLandscape3D;
