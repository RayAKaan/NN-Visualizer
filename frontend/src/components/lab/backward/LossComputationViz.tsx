import { MathRenderer } from "../MathRenderer";
import type { LossInfo } from "../../../types/pipeline";

interface Props {
  lossInfo: LossInfo;
}

export function LossComputationViz({ lossInfo }: Props) {
  const max = Math.max(...lossInfo.perClassLoss, 0.0001);
  return (
    <div className="mt-4 rounded-2xl border border-ember-700/20 bg-white p-4">
      <h3 className="text-sm font-semibold text-ink">Loss Computation</h3>
      <div className="mt-2 rounded-lg border border-barley-line bg-barley-sunken p-3">
        <MathRenderer latex="L=-\\sum_i y_i\\log(\\hat{y}_i)" displayMode />
        <MathRenderer latex={`L=-\\log(${lossInfo.predictedDistribution[lossInfo.trueLabel]?.toFixed(4) ?? "0.0000"})=${lossInfo.lossValue.toFixed(4)}`} displayMode />
      </div>
      <div className="mt-3 grid grid-cols-10 gap-1">
        {lossInfo.perClassLoss.map((v, i) => (
          <div key={i} className="text-center">
            <div className="h-14 rounded bg-barley-wash">
              <div className="w-full rounded-b"
                style={{
                  height: `${Math.max(2, (v / max) * 100)}%`,
                  background: i === lossInfo.trueLabel ? "#C2410C" : "#D8CFC0",
                }} />
            </div>
            <div className="mt-1 text-[12px] text-ink-mute">{i}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-xs text-ink-soft">
        True: {lossInfo.trueLabel} | Predicted: {lossInfo.predictedLabel} | {lossInfo.isCorrect ? "Correct" : "Incorrect"}
      </div>
    </div>
  );
}
