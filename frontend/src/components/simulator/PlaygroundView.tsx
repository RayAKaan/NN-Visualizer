import React, { useEffect, useMemo, useRef } from "react";
import { useDatasetStore } from "../../store/datasetStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { lerpColor, neuralPalette } from "@/design-system/tokens/colors";

function mix(a: string, b: string, t: number) {
  return lerpColor(a, b, Math.max(0, Math.min(1, t)));
}

export function PlaygroundView() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trainData = useDatasetStore((s) => s.trainData);

  const stats = useMemo(() => {
    if (!trainData.length) return null;
    const xs = trainData.map((p) => p.x[0]);
    const ys = trainData.map((p) => p.x[1] ?? 0);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    return { minX, maxX, minY, maxY };
  }, [trainData]);

  const centroids = useMemo(() => {
    if (!trainData.length) return null;
    const c0 = { x: 0, y: 0, n: 0 };
    const c1 = { x: 0, y: 0, n: 0 };
    trainData.forEach((p) => {
      const label = p.y?.[0] ?? 0;
      const point = label > 0 ? c1 : c0;
      point.x += p.x[0];
      point.y += p.x[1] ?? 0;
      point.n += 1;
    });
    if (c0.n) { c0.x /= c0.n; c0.y /= c0.n; }
    if (c1.n) { c1.x /= c1.n; c1.y /= c1.n; }
    return { c0, c1 };
  }, [trainData]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !stats || !centroids) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = rect.width * ratio;
    canvas.height = rect.height * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const grid = 120;
    const rangeX = stats.maxX - stats.minX || 1;
    const rangeY = stats.maxY - stats.minY || 1;
    for (let i = 0; i < grid; i += 1) {
      for (let j = 0; j < grid; j += 1) {
        const u = i / (grid - 1);
        const v = j / (grid - 1);
        const x = stats.minX + u * rangeX;
        const y = stats.minY + v * rangeY;
        const d0 = Math.hypot(x - centroids.c0.x, y - centroids.c0.y);
        const d1 = Math.hypot(x - centroids.c1.x, y - centroids.c1.y);
        const t = 1 / (1 + Math.exp((d0 - d1) * 2));
        const color = mix(neuralPalette.synapse.dim, neuralPalette.dendrite.dim, t);
        ctx.fillStyle = color;
        const px = (i / grid) * rect.width;
        const py = (j / grid) * rect.height;
        ctx.fillRect(px, py, rect.width / grid + 1, rect.height / grid + 1);
      }
    }

    // boundary highlight
    ctx.globalAlpha = 0.6;
    ctx.strokeStyle = neuralPalette.white;
    ctx.shadowColor = neuralPalette.synapse.glow;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.rect(0, 0, rect.width, rect.height);
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;

    // points
    trainData.forEach((p) => {
      const x = ((p.x[0] - stats.minX) / rangeX) * rect.width;
      const py = p.x[1] ?? 0;
      const y = ((py - stats.minY) / (stats.maxY - stats.minY || 1)) * rect.height;
      const label = p.y?.[0] ?? 0;
      const color = label > 0 ? neuralPalette.lesion.bright : neuralPalette.synapse.bright;
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = neuralPalette.void;
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }, [trainData, stats, centroids]);

  return (
    <div className="playground-view">
      <NeuralPanel className="playground-panel" variant="base">
        <div className="playground-title">Decision Boundary</div>
        {trainData.length === 0 ? (
          <div className="playground-empty">Generate a dataset to view the boundary.</div>
        ) : (
          <div className="playground-canvas-wrap">
            <canvas ref={canvasRef} className="playground-canvas" />
          </div>
        )}
      </NeuralPanel>
    </div>
  );
}
