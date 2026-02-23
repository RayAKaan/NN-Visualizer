import React, { useEffect, useRef, useState } from "react";
import { FeatureMapLayer } from "../../types";

interface Props {
  layer: FeatureMapLayer;
  maxTiles?: number;
}

function renderFM(canvas: HTMLCanvasElement, data: number[][], w: number, h: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(w, h);
  let mn = Infinity;
  let mx = -Infinity;
  for (const row of data) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const range = mx - mn || 1;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const v = ((data[y]?.[x] ?? 0) - mn) / range;
      const idx = (y * w + x) * 4;
      img.data[idx] = Math.round(10 + v * -4);
      img.data[idx + 1] = Math.round(14 + v * 168);
      img.data[idx + 2] = Math.round(23 + v * 189);
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function Tile({ data, w, h, filterIdx, mean }: { data: number[][]; w: number; h: number; filterIdx: number; mean: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [hovered, setHovered] = useState(false);

  useEffect(() => {
    if (ref.current) renderFM(ref.current, data, w, h);
  }, [data, w, h]);

  return (
    <div className="feature-map-tile" onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)} title={`Filter #${filterIdx} | Mean: ${mean.toFixed(3)}`} style={{ width: 56, height: 56 }}>
      <canvas ref={ref} width={w} height={h} style={{ width: 56, height: 56, imageRendering: "pixelated" }} />
      <div className="feature-map-label">f{filterIdx}</div>
      {hovered && <div className="fm-tooltip">Filter #{filterIdx} | Mean: {mean.toFixed(3)}</div>}
    </div>
  );
}

const FeatureMapViewer = React.memo(function FeatureMapViewer({ layer, maxTiles = 8 }: Props) {
  const indices = layer.activation_ranking.slice(0, maxTiles);
  const h = layer.shape[0] ?? 1;
  const w = layer.shape[1] ?? 1;

  return (
    <div className="feature-map-grid">
      {indices.map((filterIdx, rankIdx) => {
        const fm = layer.feature_maps[rankIdx];
        if (!fm) return null;
        return <Tile key={`${layer.layer_name}-${filterIdx}`} data={fm} w={w} h={h} filterIdx={filterIdx} mean={layer.mean_activations[filterIdx] ?? 0} />;
      })}
    </div>
  );
});

export default FeatureMapViewer;
