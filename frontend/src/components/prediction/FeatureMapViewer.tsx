import React, { useRef, useEffect, useState } from "react";
import { FeatureMapLayer } from "../../types";

interface Props {
  layer: FeatureMapLayer;
  maxTiles?: number;
}

function renderFeatureMap(canvas: HTMLCanvasElement, data: number[][], w: number, h: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(w, h);

  let min = Infinity;
  let max = -Infinity;
  for (const row of data) for (const v of row) { min = Math.min(min, v); max = Math.max(max, v); }
  const range = max - min || 1;

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const v = ((data[y]?.[x] ?? 0) - min) / range;
      const idx = (y * w + x) * 4;
      img.data[idx] = Math.round(10 + v * (6 - 10));
      img.data[idx + 1] = Math.round(14 + v * (182 - 14));
      img.data[idx + 2] = Math.round(23 + v * (212 - 23));
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function FeatureMapTile({ data, width, height, filterIndex, meanActivation }: { data: number[][]; width: number; height: number; filterIndex: number; meanActivation: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hovered, setHovered] = useState(false);

  useEffect(() => {
    if (canvasRef.current) renderFeatureMap(canvasRef.current, data, width, height);
  }, [data, width, height]);

  return (
    <div className="feature-map-tile" onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)} title={`Filter #${filterIndex} | Mean: ${meanActivation.toFixed(3)}`} style={{ width: 56, height: 56 }}>
      <canvas ref={canvasRef} width={width} height={height} style={{ width: 56, height: 56 }} />
      <div className="feature-map-label">f{filterIndex}</div>
      {hovered ? <div className="fm-tooltip">Filter #{filterIndex} | Mean: {meanActivation.toFixed(3)}</div> : null}
    </div>
  );
}

export default function FeatureMapViewer({ layer, maxTiles = 8 }: Props) {
  const indices = layer.activation_ranking.slice(0, maxTiles);
  const [h, w] = [layer.shape[0], layer.shape[1]];

  return (
    <div className="feature-map-grid">
      {indices.map((filterIdx) => {
        const fmData = layer.feature_maps[String(filterIdx)];
        if (!fmData) return null;
        return <FeatureMapTile key={filterIdx} data={fmData} width={w} height={h} filterIndex={filterIdx} meanActivation={layer.mean_activations[filterIdx] ?? 0} />;
      })}
    </div>
  );
}
