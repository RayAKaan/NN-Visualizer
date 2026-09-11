import { useExecutionStore } from "../../../store/executionStore";
import { fmtMs, opLabel, opColor } from "./visual";

export function ExecutionTimeline() {
  const trace = useExecutionStore((s) => s.trace);
  const stages = trace?.stages ?? [];
  const stageIndex = useExecutionStore((s) => s.stageIndex);
  const playing = useExecutionStore((s) => s.isPlaying);
  const play = useExecutionStore((s) => s.play);
  const pause = useExecutionStore((s) => s.pause);
  const step = useExecutionStore((s) => s.step);
  const jumpTo = useExecutionStore((s) => s.jumpTo);
  const reset = useExecutionStore((s) => s.reset);
  const speed = useExecutionStore((s) => s.speed);
  const setSpeed = useExecutionStore((s) => s.setSpeed);
  const hasInput = useExecutionStore((s) => s.hasInput);
  const totalMs = useExecutionStore((s) => s.trace?.timing.totalMs);

  if (stages.length === 0) return null;

  return (
    <div className="border-t border-barley-line bg-barley-page/80 px-4 py-2">
      <div className="mx-auto flex w-full max-w-[1120px] items-center gap-3">
        <div className="flex items-center gap-1">
          <button
            className="neural-button neural-button-ghost neural-button-sm"
            onClick={reset}
            disabled={stageIndex === 0}
            aria-label="reset to start"
          >
            ⟲
          </button>
          <button
            className="neural-button neural-button-ghost neural-button-sm"
            onClick={() => step(-1)}
            disabled={stageIndex === 0}
            aria-label="previous stage"
          >
            ‹
          </button>
          <button
            className={`neural-button neural-button-${playing ? "ghost" : "primary"} neural-button-sm ${playing ? "" : "text-white"}`}
            onClick={() => (playing ? pause() : play())}
            disabled={!hasInput}
          >
            {playing ? "⏸" : "▶"}
          </button>
          <button
            className="neural-button neural-button-ghost neural-button-sm"
            onClick={() => step(1)}
            disabled={stageIndex >= stages.length - 1}
            aria-label="next stage"
          >
            ›
          </button>
        </div>

        <input
          type="range"
          min={0}
          max={Math.max(0, stages.length - 1)}
          value={Math.min(stageIndex, stages.length - 1)}
          onChange={(e) => jumpTo(Number(e.target.value))}
          className="flex-1 accent-[var(--accent-primary)]"
          aria-label="stage position"
        />

        <div className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden" aria-label="stage chips">
          {stages.map((stage, i) => {
            const state = i < stageIndex ? "done" : i === stageIndex ? "active" : "upcoming";
            return (
              <button
                key={stage.layerId ?? stage.name ?? i}
                onClick={() => jumpTo(i)}
                title={`${stage.name} — ${opLabel(stage.operation)}`}
                className={`h-2 flex-none rounded-full transition-all ${
                  state === "active"
                    ? "w-6"
                    : state === "done"
                      ? "w-2.5"
                      : "w-2.5"
                } ${state === "active" ? "" : "hover:opacity-70"}`}
                style={{
                  background:
                    state === "done"
                      ? opColor(stage.operation)
                      : state === "active"
                        ? "var(--accent-primary)"
                        : "rgba(28,25,23,0.18)",
                }}
                aria-label={`stage ${i}: ${stage.name}`}
              />
            );
          })}
        </div>

        <div className="flex items-center gap-1">
          {[0.5, 1, 2, 4].map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`neural-button neural-button-sm ${speed === s ? "neural-button-secondary" : "neural-button-ghost"}`}
              aria-pressed={speed === s}
            >
              {s}×
            </button>
          ))}
        </div>

        <span className="shrink-0 font-mono text-[11px] text-ink-faint">
          {stageIndex + 1 < stages.length ? `${stageIndex + 1}/${stages.length}` : `${stages.length}/${stages.length}`}
          {totalMs != null ? ` · ${fmtMs(totalMs)}` : ""}
        </span>
      </div>
    </div>
  );
}