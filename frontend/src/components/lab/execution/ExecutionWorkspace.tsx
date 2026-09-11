import { useExecutionStore, selectCurrentStage } from "../../../store/executionStore";
import { useLabStore } from "../../../store/labStore";
import { ExecutionGraph } from "./ExecutionGraph";
import { ExecutionTimeline } from "./ExecutionTimeline";
import { ExecutionInspector } from "./ExecutionInspector";
import { StageDispatcher } from "./stageViews";
import { LiveFormula } from "./LiveFormula";
import { fmtMs } from "./visual";

export function ExecutionWorkspace() {
  const trace = useExecutionStore((s) => s.trace);
  const stage = useExecutionStore(selectCurrentStage);
  const phase = useExecutionStore((s) => s.cursor);
  const isPlaying = useExecutionStore((s) => s.isPlaying);
  const run = useExecutionStore((s) => s.run);
  const play = useExecutionStore((s) => s.play);
  const pause = useExecutionStore((s) => s.pause);
  const stepT = useExecutionStore((s) => s.step);
  const reset = useExecutionStore((s) => s.reset);
  const hasInput = useExecutionStore((s) => s.hasInput);

  const architecture = useLabStore((s) => s.architecture);
  const dataset = useLabStore((s) => s.dataset);
  const inputPixels = useLabStore((s) => s.inputPixels);
  const pixelCount = (inputPixels as unknown as { length?: number } | null)?.length ?? 0;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <RunBar
        architecture={architecture}
        dataset={dataset}
        pixelCount={pixelCount}
        phase={phase}
        isPlaying={isPlaying}
        hasInput={hasInput()}
        onRun={run}
        onPlay={() => (isPlaying ? pause() : play())}
        onStep={() => stepT(1)}
        onReset={reset}
      />
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(228px,270px)_minmax(0,1fr)_minmax(300px,360px)] gap-3 p-3">
        <aside className="min-h-0 overflow-y-auto rounded-2xl bg-barley-page/80 p-2">
          <ExecutionGraph />
        </aside>

        <main className="flex min-h-0 flex-col gap-3">
          {trace && stage ? (
            <>
              <StageHeader />
              <section className="min-h-0 flex-1 overflow-y-auto rounded-2xl border border-barley-line bg-barley-page/80 p-3">
                <StageDispatcher stage={stage} />
              </section>
              <LiveFormula />
            </>
          ) : (
            <EmptyState />
          )}
        </main>

        <aside className="min-h-0 overflow-y-auto rounded-2xl bg-barley-page/80 p-2">
          <ExecutionInspector />
        </aside>
      </div>
      <ExecutionTimeline />
    </div>
  );
}

function StageHeader() {
  const stage = useExecutionStore(selectCurrentStage);
  const stageIndex = useExecutionStore((s) => s.stageIndex);
  if (!stage) return null;
  const loadNote =
    stage.timing_method === "cumulative_subgraph_diff"
      ? stage.compute_time_ms != null
        ? `${fmtMs(stage.compute_time_ms)} measured`
        : "measured"
      : "";
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-2xl border border-barley-line bg-barley-page/80 px-3 py-2">
      <span className="text-[10px] font-semibold uppercase tracking-widest text-ink-faint">stage {stageIndex}</span>
      <h2 className="text-[15px] font-bold text-ink">{stage.name}</h2>
      <span
        className="rounded-full border px-2 py-0.5 text-[11px] font-semibold"
        style={{ borderColor: "var(--accent-tertiary)", color: "#065f52" }}
      >
        {stage.operation}
      </span>
      {stage.activation ? (
        <span className="rounded-full bg-barley-wash px-2 py-0.5 text-[11px] text-ink-mute">{stage.activation}</span>
      ) : null}
      <span className="text-[11px] text-ink-mute">
        {stage.inputShape.join("×")} → {stage.outputShape.join("×")}
        {stage.params != null ? ` · ${stage.params.toLocaleString()} params` : ""}
      </span>
      {loadNote ? <span className="ml-auto text-[11px] font-mono text-ink-faint">⏱ {loadNote}</span> : null}
    </div>
  );
}

function EmptyState() {
  const architecture = useLabStore((s) => s.architecture);
  const dataset = useLabStore((s) => s.dataset);
  const hasInput = useExecutionStore((s) => s.hasInput);
  return (
    <div className="grid flex-1 place-items-center">
      <div className="max-w-sm rounded-2xl border border-dashed border-barley-line bg-barley-page/50 p-6 text-center">
        <div className="text-lg font-bold text-ink">Execution workspace</div>
        <p className="mt-1 text-sm text-ink-mute">
          Run the real {architecture} weights against the current {dataset} sample. Stages stream front-to-back with
          measured timing and per-operation live math.
        </p>
        <div className="mt-2 text-[11px] text-ink-faint">topology + tensors come from the backend — never fabricated.</div>
        {!hasInput ? <div className="mt-3 text-sm font-semibold text-status-danger">Draw / pick an input sample first.</div> : null}
      </div>
    </div>
  );
}

interface RunBarProps {
  architecture: string;
  dataset: string;
  pixelCount: number;
  phase: string;
  isPlaying: boolean;
  hasInput: boolean;
  onRun: () => void;
  onPlay: () => void;
  onStep: () => void;
  onReset: () => void;
}

function RunBar({ architecture, dataset, pixelCount, phase, isPlaying, hasInput, onRun, onPlay, onStep, onReset }: RunBarProps) {
  const trace = useExecutionStore((s) => s.trace);
  const prediction = trace?.prediction;
  const busy = phase === "computing";
  return (
    <div className="lab-tools-bar mb-0">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <button onClick={onRun} disabled={!hasInput || busy} className="neural-button neural-button-primary">
          {busy ? "running…" : "▶ Run"}
        </button>
        <button onClick={onStep} disabled={!hasInput || busy} className="neural-button neural-button-ghost">step ›</button>
        <button onClick={onPlay} disabled={!hasInput || busy} className="neural-button neural-button-ghost">
          {isPlaying ? "⏸ pause" : "▶ play"}
        </button>
        <button onClick={onReset} className="neural-button neural-button-ghost">⟲ reset</button>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-mute">
        <span className="font-mono uppercase">{architecture}</span>
        <span className="text-ink-faint">·</span>
        <span>{dataset}</span>
        {pixelCount ? <span className="text-ink-faint">· {pixelCount.toLocaleString()} input units</span> : null}
        {trace ? (
          <>
            <span className="text-ink-faint">·</span>
            <span className="rounded-md bg-barley-wash px-2 py-0.5 font-mono">
              {prediction?.labelText ?? "—"} {(prediction?.probabilities ?? []).slice(0, 3).map((p) => `${(p * 100).toFixed(0)}%`).join(" / ")}
            </span>
            <span className="font-mono text-ink-faint">⏱ {fmtMs(trace.timing.totalMs)}</span>
            <span className="font-mono text-ink-faint">id {trace.execution_id.slice(0, 8)}</span>
          </>
        ) : null}
        <span className="ml-auto rounded-full border border-barley-line px-2 py-0.5 font-mono text-[10px] uppercase text-ink-faint">
          {phase}
        </span>
      </div>
    </div>
  );
}