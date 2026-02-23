import React, { useEffect, useRef } from "react";

type Props = {
  losses: number[];
  accuracies: number[];
  valLosses: number[];
  valAccuracies: number[];
};

export default function TrainingChart({ losses, accuracies, valLosses, valAccuracies }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, w, h);

    const drawLine = (arr: number[], color: string, maxY: number, yOffset: number, scale: number) => {
      if (!arr.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      arr.forEach((v, i) => {
        const x = 20 + (i / Math.max(1, arr.length - 1)) * (w - 40);
        const y = yOffset - (v / maxY) * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    const maxLoss = Math.max(0.001, ...losses, ...valLosses);
    drawLine(losses, "#06b6d4", maxLoss, h * 0.48, h * 0.38);
    drawLine(valLosses, "#f59e0b", maxLoss, h * 0.48, h * 0.38);
    drawLine(accuracies, "#22c55e", 1, h * 0.95, h * 0.38);
    drawLine(valAccuracies, "#a78bfa", 1, h * 0.95, h * 0.38);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText("Loss", 10, 14);
    ctx.fillText("Accuracy", 10, h * 0.54);
  }, [losses, accuracies, valLosses, valAccuracies]);

  return (
    <div className="card chart-container">
      <h3>📊 Training Curves</h3>
      <canvas ref={ref} width={700} height={280} className="training-graph" style={{ width: "100%", height: 280 }} />
    </div>
  );
}
