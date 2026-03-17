import React, { useEffect, useRef, useState } from "react";
import type { DatasetPoint } from "../../types/simulator";
import { neuralPalette } from "@/design-system/tokens/colors";

interface Props {
  points: DatasetPoint[];
}

export function DatasetCanvas({ points }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hovered, setHovered] = useState<{ x: number; y: number; label: number } | null>(null);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = neuralPalette.obsidian;
    ctx.fillRect(0, 0, cv.width, cv.height);
    const pad = 10;
    const xs = points.map((p) => p.x[0]);
    const ys = points.map((p) => p.x[1]);
    const minX = Math.min(...xs, -2);
    const maxX = Math.max(...xs, 2);
    const minY = Math.min(...ys, -2);
    const maxY = Math.max(...ys, 2);
    const sx = (v: number) => pad + ((v - minX) / (maxX - minX || 1)) * (cv.width - pad * 2);
    const sy = (v: number) => cv.height - pad - ((v - minY) / (maxY - minY || 1)) * (cv.height - pad * 2);

    points.forEach((p) => {
      const color = p.y[0] === 1 ? neuralPalette.synapse.bright : neuralPalette.lesion.bright;
      const cx = sx(p.x[0]);
      const cy = sy(p.x[1]);
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.shadowBlur = 4;
      ctx.shadowColor = color;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = neuralPalette.void;
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    if (hovered) {
      ctx.beginPath();
      ctx.arc(hovered.x, hovered.y, 6, 0, Math.PI * 2);
      ctx.strokeStyle = neuralPalette.pearl;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [points, hovered]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const pad = 10;
    const xs = points.map((p) => p.x[0]);
    const ys = points.map((p) => p.x[1]);
    const minX = Math.min(...xs, -2);
    const maxX = Math.max(...xs, 2);
    const minY = Math.min(...ys, -2);
    const maxY = Math.max(...ys, 2);
    const sx = (v: number) => pad + ((v - minX) / (maxX - minX || 1)) * (cv.width - pad * 2);
    const sy = (v: number) => cv.height - pad - ((v - minY) / (maxY - minY || 1)) * (cv.height - pad * 2);

    const handleMove = (e: MouseEvent) => {
      const rect = cv.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      let found: { x: number; y: number; label: number } | null = null;
      for (const p of points) {
        const cx = sx(p.x[0]);
        const cy = sy(p.x[1]);
        const dx = mx - cx;
        const dy = my - cy;
        if (Math.sqrt(dx * dx + dy * dy) <= 4) {
          found = { x: cx, y: cy, label: p.y[0] };
          break;
        }
      }
      setHovered(found);
    };

    const handleLeave = () => setHovered(null);
    cv.addEventListener("mousemove", handleMove);
    cv.addEventListener("mouseleave", handleLeave);
    return () => {
      cv.removeEventListener("mousemove", handleMove);
      cv.removeEventListener("mouseleave", handleLeave);
    };
  }, [points]);

  return (
    <div className="dataset-canvas-wrap">
      <canvas ref={canvasRef} width={280} height={220} className="dataset-canvas" />
      {hovered ? (
        <div className="dataset-tooltip" style={{ left: hovered.x + 8, top: hovered.y - 8 }}>
          ({hovered.label})
        </div>
      ) : null}
    </div>
  );
}
