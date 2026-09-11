import { useExecutionStore } from "../../../store/executionStore";
import type { Operation } from "../../../types/execution";
import { fmtMs, opColor, opLabel } from "./visual";

export function ExecutionGraph() {
  const stageIndex = useExecutionStore((s) => s.stageIndex);
  const trace = useExecutionStore((s) => s.trace);
  const jumpTo = useExecutionStore((s) => s.jumpTo);
  const stages = trace?.stages ?? [];
  const model = trace?.model;

  return (
    <div className="flex min-w-0 flex-col gap-1.5 py-1">
      <div className="px-1 text-[10px] font-semibold uppercase tracking-widest text-ink-faint">
        graph · backend topology
      </div>
      <div className="flex items-center gap-2 px-1 pb-1">
        <span
          className="rounded-full border px-2 py-0.5 text-[11px] font-semibold"
          style={{ borderColor: opColor("input"), color: "var(--accent-secondary)" }}
        >
          {model?.id ?? "model"} · {stages.length} ops
        </span>
        <span className="text-[11px] text-ink-faint">click a node to jump</span>
      </div>

      <div className="relative pl-4">
        <div className="absolute bottom-2 left-[22px] top-2 w-px bg-barley-line" />
        {stages.map((stage, i) => {
          const isActive = i === stageIndex;
          const isDone = i < stageIndex;
          const color = opColor(stage.operation);
          return (
            <button
              key={stage.layerId ?? stage.name ?? i}
              onClick={() => jumpTo(i)}
              className={`relative mb-1 flex w-full min-w-0 items-center gap-2 rounded-xl border px-2 py-1.5 text-left transition-colors ${
                isActive
                  ? "border-[var(--accent-primary)] bg-[color-mix(in_srgb,var(--accent-primary)_8%,white)]"
                  : "border-barley-line bg-barley-page/70 hover:bg-barley-wash"
              }`}
              style={isActive ? { boxShadow: "0 0 0 1px var(--accent-primary) inset" } : undefined}
              aria-pressed={isActive}
            >
              <span
                className="grid h-6 w-6 shrink-0 place-items-center rounded-full text-[11px] font-bold text-white"
                style={{ background: color, opacity: isDone ? 0.55 : 1 }}
              >
                {isDone ? "✓" : i}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13px] font-semibold leading-tight text-ink">
                  {stage.name}
                </span>
                <span className="block text-[11px] leading-tight text-ink-mute">
                  {opLabel(stage.operation)} · {stage.outputShape.join("×")}
                </span>
              </span>
              <span className="shrink-0 text-right font-mono text-[10px] text-ink-faint">
                {stage.compute_time_ms != null ? fmtMs(stage.compute_time_ms) : "—"}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export const OPERATION_SAMPLES: Record<Operation, string> = {
  input: "signal in",
  dense: "y = Wx + b",
  conv2d: "Σ x·k + b",
  max_pool: "max 2×2",
  avg_pool: "avg 2×2",
  flatten: "reshape",
  activation: "σ(x)",
  lstm: "cell gates",
  lstm_bidirectional: "cell gates",
  dropout: "y = x",
  softmax: "p(k)",
};