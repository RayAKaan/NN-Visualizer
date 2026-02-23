import React from "react";
import { KernelLayer } from "../../types";

interface Props {
  layer: KernelLayer;
  maxKernels?: number;
}

export default function KernelViewer({ layer, maxKernels = 6 }: Props) {
  const keys = Object.keys(layer.kernels).slice(0, maxKernels);

  return (
    <div className="feature-map-grid">
      {keys.map((k) => {
        const kernel = layer.kernels[k];
        const h = kernel.length;
        const w = kernel[0]?.length ?? 0;
        const values = kernel.flat();
        const min = Math.min(...values, 0);
        const max = Math.max(...values, 1);
        const range = max - min || 1;

        return (
          <div key={k} className="layer-block">
            <div className="kernel-grid" style={{ gridTemplateColumns: `repeat(${w}, 16px)` }}>
              {kernel.flat().map((val, idx) => {
                const t = (val - min) / range;
                const isPos = val >= 0;
                const color = isPos
                  ? `rgba(6,182,212,${0.2 + t * 0.8})`
                  : `rgba(239,68,68,${0.2 + (1 - t) * 0.8})`;
                return <div key={idx} className="kernel-cell" style={{ backgroundColor: color }} title={`k${k}[${Math.floor(idx / w)},${idx % w}] = ${val.toFixed(3)}`} />;
              })}
            </div>
            <span className="layer-label">k{k} ({h}×{w})</span>
          </div>
        );
      })}
    </div>
  );
}
