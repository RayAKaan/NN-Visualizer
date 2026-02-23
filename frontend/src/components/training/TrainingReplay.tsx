import React, { useMemo, useState } from "react";

type Props = {
  losses: number[];
  accuracies: number[];
};

export default function TrainingReplay({ losses, accuracies }: Props) {
  const [idx, setIdx] = useState(0);
  const total = Math.max(losses.length, accuracies.length);
  const safeIdx = Math.min(idx, Math.max(0, total - 1));
  const summary = useMemo(() => ({ loss: losses[safeIdx] ?? 0, acc: accuracies[safeIdx] ?? 0 }), [losses, accuracies, safeIdx]);

  return (
    <div className="card timeline-container">
      <h3>⏱️ Training Replay</h3>
      <input className="timeline-slider" type="range" min={0} max={Math.max(0, total - 1)} value={safeIdx} onChange={(e) => setIdx(Number(e.target.value))} style={{ width: "100%" }} />
      <div className="timeline-info">Step {safeIdx + 1}/{Math.max(total, 1)} · Loss {summary.loss.toFixed(4)} · Acc {(summary.acc * 100).toFixed(2)}%</div>
    </div>
  );
}
