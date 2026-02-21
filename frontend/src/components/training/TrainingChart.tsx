import React, { useMemo } from "react";

interface TrainingChartProps {
  title: string;
  values: number[];
  height?: number;
  color?: string;
}

const TrainingChart: React.FC<TrainingChartProps> = ({ title, values, height = 120, color = "#4c1d95" }) => {
  const points = useMemo(() => {
    if (values.length === 0) return "";
    const max = Math.max(...values);
    const min = Math.min(...values);
    const range = max - min || 1;
    return values
      .map((value, index) => {
        const x = (index / Math.max(1, values.length - 1)) * 100;
        const y = 100 - ((value - min) / range) * 100;
        return `${x},${y}`;
      })
      .join(" ");
  }, [values]);

  return (
    <div className="chart">
      <div className="chart-title">{title}</div>
      <svg viewBox="0 0 100 100" style={{ height }}>
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="2"
          points={points}
        />
      </svg>
    </div>
  );
};

export default TrainingChart;
