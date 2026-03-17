import React, { useEffect, useMemo, useRef, useState } from "react";

export interface GlowLineChartProps {
  values: number[];
  color: string;
  label: string;
  height?: number;
}

export function GlowLineChart({ values, color, label, height = 180 }: GlowLineChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const minMax = useMemo(() => {
    if (values.length === 0) return { min: 0, max: 1 };
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) return { min: min - 1, max: max + 1 };
    return { min, max };
  }, [values]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapRef.current;
    if (!canvas || !wrapper) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const render = () => {
      const rect = wrapper.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = rect.width * ratio;
      canvas.height = height * ratio;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${height}px`;

      const w = rect.width;
      const h = height;
      ctx.save();
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // grid
      ctx.strokeStyle = "rgba(36,40,54,0.12)";
      ctx.lineWidth = 1;
      for (let i = 1; i <= 5; i += 1) {
        const y = (h / 6) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      if (values.length > 1) {
        const range = minMax.max - minMax.min;
        const toY = (v: number) => h - ((v - minMax.min) / range) * (h - 20) - 10;
        const toX = (idx: number) => (idx / (values.length - 1)) * (w - 20) + 10;

        // glow
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 6;
        ctx.globalAlpha = 0.2;
        ctx.shadowColor = color;
        ctx.shadowBlur = 16;
        ctx.beginPath();
        values.forEach((v, i) => {
          const x = toX(i);
          const y = toY(v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.restore();

        // main line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        values.forEach((v, i) => {
          const x = toX(i);
          const y = toY(v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // fill
        const fill = ctx.createLinearGradient(0, 0, 0, h);
        fill.addColorStop(0, `${color}22`);
        fill.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = fill;
        ctx.beginPath();
        values.forEach((v, i) => {
          const x = toX(i);
          const y = toY(v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.lineTo(w - 10, h - 10);
        ctx.lineTo(10, h - 10);
        ctx.closePath();
        ctx.fill();

        // endpoint
        const lastIdx = values.length - 1;
        const endX = toX(lastIdx);
        const endY = toY(values[lastIdx]);
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;
        ctx.arc(endX, endY, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // hover crosshair
        if (hoverX != null) {
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = "rgba(36,40,54,0.6)";
          ctx.beginPath();
          ctx.moveTo(hoverX, 6);
          ctx.lineTo(hoverX, h - 6);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      ctx.restore();
    };

    render();
    const observer = new ResizeObserver(render);
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, [values, color, height, minMax, hoverX]);

  return (
    <div className="glow-chart" ref={wrapRef}>
      <div className="glow-chart-header">{label}</div>
      <canvas
        ref={canvasRef}
        className="glow-chart-canvas"
        onMouseMove={(e) => {
          const rect = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect();
          setHoverX(e.clientX - rect.left);
        }}
        onMouseLeave={() => setHoverX(null)}
      />
    </div>
  );
}
