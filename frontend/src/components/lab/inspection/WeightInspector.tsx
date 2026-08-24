import { useMemo } from "react";
import type { WeightInspectionData } from "../../../types/pipeline";
import { renderHeatmap, WEIGHT_RAMP_LIGHT } from "../../../utils/colorRamps";
import { WeightDistribution } from "./WeightDistribution";

interface Props {
  data: WeightInspectionData;
  onClose: () => void;
}

export function WeightInspector({ data, onClose }: Props) {
  const ramp = WEIGHT_RAMP_LIGHT;
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
    <div className="fixed right-4 top-20 bottom-24 z-50 w-[380px] overflow-y-auto rounded-2xl border border-ink/10 bg-barley-page/92 p-4 shadow-pop backdrop-blur-xl transition-all duration-300">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold text-ink">Weight Inspector</div>
          <div className="text-xs text-ink-mute">{data.stageId}</div>
        </div>
        <button type="button" onClick={onClose} className="rounded-md border border-barley-linestrong bg-white px-2 py-1 text-xs text-ink-soft hover:border-ember-600/40">Close</button>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded border border-barley-linestrong bg-white p-2">Mean {data.statistics.mean.toFixed(4)}</div>
        <div className="rounded border border-barley-linestrong bg-white p-2">Std {data.statistics.std.toFixed(4)}</div>
        <div className="rounded border border-barley-linestrong bg-white p-2">Min {data.statistics.min.toFixed(4)}</div>
        <div className="rounded border border-barley-linestrong bg-white p-2">Max {data.statistics.max.toFixed(4)}</div>
      </div>

      <div className="rounded border border-barley-linestrong bg-white p-2">
        <img src={heat} alt="weight heatmap" className="w-full rounded" style={{ imageRendering: "pixelated" }} />
      </div>

      <div className="mt-3">
        <WeightDistribution bins={data.statistics.distribution.bins} counts={data.statistics.distribution.counts} />
      </div>

      {data.untrainedStatistics ? (
        <div className="mt-3 rounded border border-status-warning/30 bg-status-warning/10 p-2 text-xs">
          Untrained std: {data.untrainedStatistics.std.toFixed(4)} | Trained std: {data.statistics.std.toFixed(4)}
        </div>
      ) : null}
    </div>
  );
}
