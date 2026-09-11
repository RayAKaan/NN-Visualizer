import { memo } from "react";
import type { Operation, TraceStageStatistics } from "../../../types/execution";

export const OP_LABEL: Record<string, string> = {
  input: "Input",
  dense: "Dense",
  conv2d: "Conv2D",
  max_pool: "MaxPool",
  avg_pool: "AvgPool",
  flatten: "Flatten",
  activation: "Activation",
  lstm: "LSTM",
  lstm_bidirectional: "BiLSTM",
  dropout: "Dropout",
  softmax: "Softmax",
};

export function opLabel(op: Operation): string {
  return OP_LABEL[op] ?? op;
}

const COLORS: Record<string, string> = {
  input: "#0072b2",
  dense: "#c2410c",
  conv2d: "#00806a",
  max_pool: "#8a5a2b",
  avg_pool: "#8a5a2b",
  flatten: "#7c6a8f",
  activation: "#7c6a8f",
  lstm: "#a64d85",
  lstm_bidirectional: "#a64d85",
  dropout: "#57534e",
  softmax: "#b45309",
};

export function opColor(op: Operation): string {
  return COLORS[op] ?? "#57534e";
}

export function boxShadowFor(op: Operation, active: boolean): React.CSSProperties {
  const c = opColor(op);
  return active
    ? { boxShadow: `0 0 0 2px #ffffff, 0 0 0 4.5px ${c}` }
    : { borderColor: "rgba(28,25,23,0.14)" };
}

export function fmt(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return "—";
  if (n === 0) return "0";
  const a = Math.abs(n);
  if (a >= 1e6) return n.toExponential(2);
  if (a < 1e-3) return n.toExponential(2);
  return Number(n.toFixed(digits)).toString();
}

export function fmtMs(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  if (ms < 0.001) return `${(ms * 1e6).toFixed(0)}ns`;
  if (ms < 1) return `${(ms * 1000).toFixed(1)}µs`;
  const seconds = ms / 1000;
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  if (seconds >= 1) return `${seconds.toFixed(2)}s`;
  return `${ms.toFixed(2)}ms`;
}

function colorForValue(v: number, absMax: number): string {
  if (v === 0 || absMax <= 0) return "rgba(236,231,222,1)";
  const alpha = Math.min(1, Math.abs(v) / absMax);
  if (v > 0) return `rgba(194,65,12,${0.08 + alpha * 0.9})`;
  return `rgba(0,114,178,${0.08 + alpha * 0.9})`;
}

interface HeatmapProps {
  map: number[][];
  absMax?: number;
  cell?: number;
  onCell?: (h: number, w: number, value: number) => void;
  ariaLabel?: string;
}

export const TensorHeatmap = memo(function TensorHeatmap({ map, absMax, cell = 18, onCell, ariaLabel }: HeatmapProps) {
  const max = absMax ?? Math.max(
    ...map.flat().map((v) => Math.abs(v)),
    1e-6,
  );
  const h = map.length;
  const w = h > 0 ? map[0].length : 0;
  return (
    <div
      role="img"
      aria-label={ariaLabel ?? `feature map ${h}×${w}`}
      className="grid gap-px"
      style={{ gridTemplateColumns: `repeat(${w}, ${cell}px)`, width: w * (cell + 1) - 1 }}
    >
      {map.map((row, r) =>
        row.map((v, c) => {
          const bg = v === 0 ? "rgba(236,231,222,0.55)" : colorForValue(v, max);
          const content = (
            <div
              key={`${r}-${c}`}
              onPointerEnter={onCell ? () => onCell(r, c, v) : undefined}
              onClick={onCell ? (e) => { e.stopPropagation(); onCell(r, c, v); } : undefined}
              className={onCell ? "cursor-crosshair" : undefined}
              title={`(${r},${c}) = ${fmt(v)}`}
              style={{ width: cell, height: cell, background: bg }}
            />
          );
          return content;
        }),
      )}
    </div>
  );
});

interface StripProps {
  values: number[];
  maxBarCount?: number;
  height?: number;
  signedPositiveColor?: string;
  signedNegativeColor?: string;
  onClick?: (index: number, value: number) => void;
  highlightIndex?: number;
  title?: string;
}

export const TensorStrip = memo(function TensorStrip({
  values,
  maxBarCount = 192,
  height = 26,
  signedPositiveColor = "rgba(194,65,12,0.85)",
  signedNegativeColor = "rgba(0,114,178,0.85)",
  onClick,
  highlightIndex,
  title,
}: StripProps) {
  const barCount = Math.min(values.length, maxBarCount);
  const stride = Math.ceil(values.length / barCount);
  const indices = Array.from({ length: barCount }, (_, i) => Math.min(i * stride, values.length - 1));
  const maxAbs = Math.max(...indices.map((idx) => Math.abs(values[idx])), 1e-6);
  return (
    <div
      className="flex w-full overflow-hidden rounded-md"
      style={{ height, gap: 1, background: "rgba(28,25,23,0.05)" }}
      role="img"
      aria-label={title ?? `${values.length} activation values`}
    >
      {indices.map((sampleIdx, barIdx) => {
        const v = values[sampleIdx];
        const pct = Math.max(8, (Math.abs(v) / maxAbs) * 100);
        const bg = v >= 0 ? signedPositiveColor : signedNegativeColor;
        const isHi = highlightIndex != null && sampleIdx === highlightIndex;
        return (
          <div
            key={`bar-${barIdx}`}
            className="relative"
            style={{ flex: "1 1 0", height: "100%", cursor: onClick ? "pointer" : undefined }}
            onClick={onClick ? () => onClick(sampleIdx, v) : undefined}
            title={`i=${sampleIdx} → ${fmt(v)}`}
          >
            <div className="absolute bottom-0 left-0 right-0" style={{ height: `${pct}%`, background: isHi ? "#1c1917" : bg }} />
            {isHi ? <div className="absolute inset-0 ring-1 ring-inset ring-ink" /> : null}
          </div>
        );
      })}
    </div>
  );
});

export function StatChips({ stats }: { stats: TraceStageStatistics }) {
  const items = [
    ["min", fmt(stats.min)],
    ["max", fmt(stats.max)],
    ["mean", fmt(stats.mean, 3)],
    ["std", fmt(stats.std, 3)],
    ["sparsity", `${(stats.sparsity * 100).toFixed(1)}%`],
    ["zeros", `${(stats.zero * 100).toFixed(1)}%`],
  ];
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map(([k, v]) => (
        <span key={k} className="rounded-md border border-barley-line bg-barley-wash px-1.5 py-0.5 text-[11px] text-ink-mute">
          <span className="font-semibold text-ink-soft">{k}</span> {v}
        </span>
      ))}
    </div>
  );
}

export function TruthChip({ ok, label, detail }: { ok: boolean; label: string; detail?: string }) {
  return (
    <span
      title={detail}
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold ${
        ok ? "border-status-success text-status-success" : "border-status-danger text-status-danger"
      }`}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: ok ? "var(--success)" : "var(--danger)" }} />
      {label}
    </span>
  );
}