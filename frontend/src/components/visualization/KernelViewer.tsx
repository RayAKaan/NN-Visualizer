import React from "react";

interface Props {
  kernel: number[][];
  filterIndex: number;
}

function kernelColor(v: number, absMax: number): string {
  const range = absMax || 1;
  const norm = v / range;
  if (norm < 0) {
    const t = Math.abs(norm);
    return `rgb(${Math.round(26 + t * 113)},${Math.round(32 + t * 60)},${Math.round(53 + t * 193)})`;
  }
  const t = norm;
  return `rgb(${Math.round(26 - t * 20)},${Math.round(32 + t * 150)},${Math.round(53 + t * 159)})`;
}

const KernelViewer = React.memo(function KernelViewer({ kernel, filterIndex }: Props) {
  let absMax = 0;
  for (const row of kernel) for (const v of row) if (Math.abs(v) > absMax) absMax = Math.abs(v);
  const cols = kernel[0]?.length || 0;

  return (
    <div style={{ display: "inline-block" }}>
      <div className="kernel-grid" style={{ gridTemplateColumns: `repeat(${cols}, 16px)` }}>
        {kernel.flatMap((row, y) => row.map((v, x) => (
          <div key={`${y}-${x}`} className="kernel-cell" title={`[${y},${x}] = ${v.toFixed(4)}`} style={{ backgroundColor: kernelColor(v, absMax), width: 16, height: 16 }} />
        )))}
      </div>
      <div style={{ fontSize: 9, textAlign: "center", color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginTop: 2 }}>K#{filterIndex}</div>
    </div>
  );
});

export default KernelViewer;
