import React, { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useNeurofluxStore } from "../../store/useNeurofluxStore";

export default function MetricsDashboard() {
  const metricsHistory = useNeurofluxStore((s) => s.metricsHistory);
  const currentEpoch = useNeurofluxStore((s) => s.currentEpoch);
  const scrubToEpoch = useNeurofluxStore((s) => s.scrubToEpoch);
  const safeMetrics = useMemo(
    () =>
      metricsHistory.filter(
        (m) =>
          Number.isFinite(m?.epoch) &&
          Number.isFinite(m?.loss) &&
          Number.isFinite(m?.accuracy),
      ),
    [metricsHistory],
  );

  const onChartClick = (state: any) => {
    const label = state?.activeLabel;
    if (typeof label === "number") {
      scrubToEpoch(label);
    }
  };

  const currentPoint = useMemo(() => {
    if (safeMetrics.length === 0) return null;
    const byEpoch = safeMetrics.find((m) => m.epoch === currentEpoch);
    if (byEpoch) return byEpoch;
    const idx = Math.max(0, Math.min(safeMetrics.length - 1, currentEpoch));
    return safeMetrics[idx] ?? null;
  }, [safeMetrics, currentEpoch]);

  const landscapePos = useMemo(() => {
    if (!currentPoint || safeMetrics.length <= 1) return { left: 12, top: 60 };
    const minLoss = Math.min(...safeMetrics.map((m) => m.loss));
    const maxLoss = Math.max(...safeMetrics.map((m) => m.loss));
    const lossRange = Math.max(0.0001, maxLoss - minLoss);
    const left = (currentPoint.epoch / (safeMetrics[safeMetrics.length - 1].epoch || 1)) * 96;
    const top = (1 - (currentPoint.loss - minLoss) / lossRange) * 72 + 8;
    return { left, top };
  }, [safeMetrics, currentPoint]);

  return (
    <div className="h-full rounded-xl border border-slate-700 bg-slate-900/85 p-3 overflow-hidden">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-cyan-300">Metrics Dashboard</h3>
        <div className="text-xs text-slate-400">Epoch {currentEpoch}</div>
      </div>

      <div className="relative h-[220px] rounded-lg overflow-hidden border border-slate-700/60 metrics-landscape-wrap">
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_20%_25%,rgba(236,72,153,0.18),transparent_45%),radial-gradient(circle_at_78%_70%,rgba(59,130,246,0.16),transparent_42%),linear-gradient(145deg,rgba(15,23,42,0.94),rgba(2,6,23,0.95))] metrics-landscape-bg" />
        {safeMetrics.length === 0 && (
          <div className="absolute inset-0 z-20 flex items-center justify-center text-xs font-mono text-slate-400">
            Waiting for training metrics stream...
          </div>
        )}
        <div
          className="absolute w-4 h-4 rounded-full bg-cyan-300 shadow-[0_0_18px_6px_rgba(34,211,238,0.45)] pointer-events-none"
          style={{ left: `calc(${landscapePos.left}% - 8px)`, top: `calc(${landscapePos.top}% - 8px)` }}
        />
        <div className="absolute inset-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={safeMetrics} onClick={onChartClick}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="epoch" stroke="#94a3b8" />
              <YAxis yAxisId="loss" stroke="#94a3b8" />
              <YAxis yAxisId="acc" orientation="right" domain={[0, 1]} stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155" }} />
              <Legend />
              <ReferenceLine x={currentEpoch} yAxisId="loss" stroke="#a78bfa" strokeDasharray="4 4" />
              <Line yAxisId="loss" type="monotone" dataKey="loss" stroke="#f43f5e" dot={false} strokeWidth={2} />
              <Line yAxisId="acc" type="monotone" dataKey="accuracy" stroke="#22c55e" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
