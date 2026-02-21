import React, { useEffect, useMemo, useRef, useState } from "react";
import { FeatureMapLayer, KernelLayer } from "../../types";
import KernelViewer from "./KernelViewer";

const renderFeatureMap = (
  canvas: HTMLCanvasElement,
  data: number[][],
  width: number,
  height: number
) => {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const imageData = ctx.createImageData(width, height);
  let min = Infinity;
  let max = -Infinity;
  for (const row of data) {
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const range = max - min || 1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const v = (data[y][x] - min) / range;
      const i = (y * width + x) * 4;
      imageData.data[i] = Math.round(v * 6);
      imageData.data[i + 1] = Math.round(v * 182);
      imageData.data[i + 2] = Math.round(v * 212);
      imageData.data[i + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
};

interface TileProps {
  idx: string;
  data: number[][];
  mean: number;
}

const FeatureMapTile: React.FC<TileProps> = ({ idx, data, mean }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState(false);

  useEffect(() => {
    if (!canvasRef.current) return;
    renderFeatureMap(canvasRef.current, data, data[0]?.length ?? 1, data.length ?? 1);
  }, [data]);

  return (
    <div className="feature-map-tile" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      {hover && <div className="fm-tooltip">Filter #{idx} | Mean: {mean.toFixed(3)}</div>}
      <canvas ref={canvasRef} width={data[0]?.length ?? 1} height={data.length ?? 1} style={{ width: 56, height: 56 }} />
      <div className="feature-map-label">f{idx}</div>
    </div>
  );
};

const FeatureMapViewer: React.FC<{ featureMaps: FeatureMapLayer[]; kernels: KernelLayer[] }> = ({ featureMaps, kernels }) => {
  const kernelMap = useMemo(() => {
    const map: Record<string, Record<string, number[][]>> = {};
    kernels.forEach((k) => { map[k.layer_name] = k.kernels; });
    return map;
  }, [kernels]);

  return (
    <div className="cnn-flow">
      {featureMaps.map((layer, i) => (
        <React.Fragment key={layer.layer_name}>
          {i > 0 && <div className="cnn-flow-arrow"><span>{layer.layer_type === "pool" ? "2×2 max pool" : "3×3 conv"}</span></div>}
          <div className="cnn-layer-block">
            <div className="layer-title">{layer.layer_name} · {layer.shape[0]}×{layer.shape[1]}×{layer.shape[2]}</div>
            <div className="feature-map-grid">
              {layer.activation_ranking.map((filterIdx) => (
                <FeatureMapTile
                  key={`${layer.layer_name}-${filterIdx}`}
                  idx={String(filterIdx)}
                  data={layer.feature_maps[String(filterIdx)]}
                  mean={layer.mean_activations[filterIdx] ?? 0}
                />
              ))}
            </div>
            {layer.layer_type === "conv" && <KernelViewer kernels={kernelMap[layer.layer_name] ?? {}} />}
          </div>
        </React.Fragment>
      ))}
    </div>
  );
};

export default FeatureMapViewer;
