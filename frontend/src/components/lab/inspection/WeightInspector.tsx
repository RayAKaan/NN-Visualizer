import { useMemo } from "react";
import type { WeightInspectionData } from "../../../types/pipeline";
import { renderHeatmap, WEIGHT_RAMP_DARK, WEIGHT_RAMP_LIGHT } from "../../../utils/colorRamps";
import { WeightDistribution } from "./WeightDistribution";
import { useThemeColors } from "../../../hooks/useThemeColors";

interface Props {
  data: WeightInspectionData;
  onClose: () => void;
}

export function WeightInspector({ data, onClose }: Props) {
  const { isDark } = useThemeColors();
  const ramp = isDark ? WEIGHT_RAMP_DARK : WEIGHT_RAMP_LIGHT;
  const rows = data.shape[0] ?? 1;
  const cols = data.shape[1] ?? 1;
  const showRows = Math.min(rows, 32);
  const showCols = Math.min(cols, 32);

  const heat = useMemo(() => {
    const src = data.weights;
    const out = new Float32Array(showRows * showCols);
    for (let r = 0; r < showRows; r += 1) {
      for (let c = 0; c < showCols; c += 1) {
        const idx = r * cols + c;
        const v = src[idx] ?? 0;
        out[r * showCols + c] = (v - data.statistics.min) / ((data.statistics.max - data.statistics.min) || 1);
      }
    }
    return renderHeatmap(out, showCols, showRows, ramp, false);
  }, [cols, data.statistics.max, data.statistics.min, data.weights, ramp, showCols, showRows]);

  return (
    <div className="fixed right-4 top-20 bottom-24 z-50 w-[380px] overflow-y-auto rounded-2xl border p-4 shadow-2xl backdrop-blur-xl transition-all duration-300" style={{ background: "rgba(18,20,28,0.85)", borderColor: "rgba(59,130,246,0.2)" }}>
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>Weight Inspector</div>
          <div className="text-xs" style={{ color: "var(--text-3)" }}>{data.stageId}</div>
        </div>
        <button type="button" onClick={onClose} className="rounded-md px-2 py-1 text-xs" style={{ background: "var(--bg-panel)", color: "var(--text-2)" }}>Close</button>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>Mean {data.statistics.mean.toFixed(4)}</div>
        <div className="rounded border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>Std {data.statistics.std.toFixed(4)}</div>
        <div className="rounded border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>Min {data.statistics.min.toFixed(4)}</div>
        <div className="rounded border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>Max {data.statistics.max.toFixed(4)}</div>
      </div>

      <div className="rounded border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
        <img src={heat} alt="weight heatmap" className="w-full rounded" style={{ imageRendering: "pixelated" }} />
      </div>

      <div className="mt-3">
        <WeightDistribution bins={data.statistics.distribution.bins} counts={data.statistics.distribution.counts} />
      </div>

      {data.untrainedStatistics ? (
        <div className="mt-3 rounded border p-2 text-xs" style={{ borderColor: "rgba(251,191,36,0.3)", background: "var(--status-warning-bg)" }}>
          Untrained std: {data.untrainedStatistics.std.toFixed(4)} | Trained std: {data.statistics.std.toFixed(4)}
        </div>
      ) : null}
    </div>
  );
}
