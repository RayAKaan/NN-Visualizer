import React from "react";

type Props = {
  matrix: number[][] | null;
};

export default function ConfusionMatrix({ matrix }: Props) {
  if (!matrix || !matrix.length) {
    return <div className="card"><h3>🔢 Confusion Matrix</h3><div className="stub-placeholder">No epoch metrics yet</div></div>;
  }

  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="card">
      <h3>🔢 Confusion Matrix</h3>
      <div className="confusion-matrix-grid" style={{ display: "grid", gridTemplateColumns: `repeat(${matrix[0].length}, 1fr)`, gap: 2 }}>
        {matrix.flatMap((row, i) => row.map((v, j) => (
          <div key={`${i}-${j}`} className="confusion-cell" title={`true ${i}, pred ${j}, count ${v}`} style={{ padding: 6, textAlign: "center", background: `rgba(34,197,94,${0.1 + (v / max) * 0.8})` }}>
            {v}
          </div>
        )))}
      </div>
    </div>
  );
}
