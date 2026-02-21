import React from "react";

interface Props {
  kernels: Record<string, number[][]>;
}

const KernelViewer: React.FC<Props> = ({ kernels }) => {
  const items = Object.entries(kernels);
  if (!items.length) return null;

  return (
    <div className="feature-map-grid">
      {items.map(([idx, kernel]) => {
        const flat = kernel.flat();
        const maxAbs = Math.max(...flat.map((v) => Math.abs(v)), 1e-6);
        return (
          <div key={idx} className="feature-map-tile" style={{ padding: 6 }}>
            <div className="kernel-grid" style={{ gridTemplateColumns: `repeat(${kernel[0]?.length ?? 0}, 16px)` }}>
              {kernel.flatMap((row, y) =>
                row.map((v, x) => {
                  const n = v / maxAbs;
                  const color = n >= 0
                    ? `rgba(6,182,212,${Math.abs(n)})`
                    : `rgba(139,92,246,${Math.abs(n)})`;
                  return <div key={`${idx}-${x}-${y}`} className="kernel-cell" style={{ background: color }} />;
                })
              )}
            </div>
            <div className="feature-map-label">Kernel #{idx}</div>
          </div>
        );
      })}
    </div>
  );
};

export default KernelViewer;
