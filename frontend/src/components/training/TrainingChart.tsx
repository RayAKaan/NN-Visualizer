import React, { useRef, useEffect } from "react";

interface OverlayPoint {
  index: number;
  value: number;
  color: string;
}

interface Props {
  data: number[];
  label: string;
  color: string;
  height?: number;
  maxPoints?: number;
  overlayPoints?: OverlayPoint[];
  minY?: number;
  maxY?: number;
}

export default function TrainingChart({
  data,
  label,
  color,
  height = 120,
  maxPoints = 300,
  overlayPoints,
  minY,
  maxY,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 20, right: 10, bottom: 20, left: 50 };

    ctx.fillStyle = "#1a2035";
    ctx.fillRect(0, 0, w, h);

    const visibleData = data.slice(-maxPoints);
    if (visibleData.length < 2) {
      ctx.fillStyle = "#64748b";
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for data...", w / 2, h / 2);
      return;
    }

    let yMin = minY ?? Math.min(...visibleData);
    let yMax = maxY ?? Math.max(...visibleData);
    if (yMin === yMax) {
      yMin -= 0.1;
      yMax += 0.1;
    }

    const yRange = yMax - yMin;
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    const toX = (i: number) => pad.left + (i / (visibleData.length - 1)) * plotW;
    const toY = (v: number) => pad.top + (1 - (v - yMin) / yRange) * plotH;

    ctx.strokeStyle = "#2a3555";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i += 1) {
      const y = pad.top + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();

      const val = yMax - (i / 4) * yRange;
      ctx.fillStyle = "#64748b";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(val.toFixed(3), pad.left - 4, y + 3);
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    visibleData.forEach((v, i) => {
      const x = toX(i);
      const y = toY(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const gradient = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
    gradient.addColorStop(0, `${color}40`);
    gradient.addColorStop(1, `${color}05`);
    ctx.fillStyle = gradient;
    ctx.beginPath();
    visibleData.forEach((v, i) => {
      const x = toX(i);
      const y = toY(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(toX(visibleData.length - 1), h - pad.bottom);
    ctx.lineTo(toX(0), h - pad.bottom);
    ctx.closePath();
    ctx.fill();

    if (overlayPoints) {
      for (const pt of overlayPoints) {
        const adjustedIdx = pt.index - (data.length - visibleData.length);
        if (adjustedIdx < 0 || adjustedIdx >= visibleData.length) continue;
        const x = toX(adjustedIdx);
        const y = toY(pt.value);
        ctx.fillStyle = pt.color;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0a0e17";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    const lastVal = visibleData[visibleData.length - 1];
    ctx.fillStyle = color;
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText(lastVal.toFixed(4), w - pad.right, pad.top - 6);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText(label, pad.left, pad.top - 6);
  }, [data, color, label, height, maxPoints, overlayPoints, minY, maxY]);

  return (
    <div className="chart-container">
      <canvas ref={canvasRef} width={400} height={height} style={{ width: "100%", height }} />
    </div>
  );
}
