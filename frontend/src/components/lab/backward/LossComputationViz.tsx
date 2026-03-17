import { MathRenderer } from "../MathRenderer";
import type { LossInfo } from "../../../types/pipeline";

interface Props {
  lossInfo: LossInfo;
}

export function LossComputationViz({ lossInfo }: Props) {
  const max = Math.max(...lossInfo.perClassLoss, 0.0001);
  return (
    <div className="mt-4 rounded-2xl border p-4" style={{ borderColor: "var(--bwd-border)", background: "var(--bg-card)" }}>
      <h3 className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>Loss Computation</h3>
      <div className="mt-2 rounded-lg p-3" style={{ background: "var(--math-panel-bg)", border: "1px solid var(--math-panel-border)" }}>
        <MathRenderer latex="L=-\\sum_i y_i\\log(\\hat{y}_i)" displayMode />
        <MathRenderer latex={`L=-\\log(${lossInfo.predictedDistribution[lossInfo.trueLabel]?.toFixed(4) ?? "0.0000"})=${lossInfo.lossValue.toFixed(4)}`} displayMode />
      </div>
      <div className="mt-3 grid grid-cols-10 gap-1">
        {lossInfo.perClassLoss.map((v, i) => (
          <div key={i} className="text-center">
            <div className="h-14 rounded bg-slate-800/50">
              <div className="w-full rounded-b"
                style={{
                  height: `${Math.max(2, (v / max) * 100)}%`,
                  background: i === lossInfo.trueLabel ? "var(--bwd)" : "var(--text-4)",
                }} />
            </div>
            <div className="mt-1 text-[10px]" style={{ color: "var(--text-3)" }}>{i}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-xs" style={{ color: "var(--text-2)" }}>
        True: {lossInfo.trueLabel} | Predicted: {lossInfo.predictedLabel} | {lossInfo.isCorrect ? "Correct" : "Incorrect"}
      </div>
    </div>
  );
}
