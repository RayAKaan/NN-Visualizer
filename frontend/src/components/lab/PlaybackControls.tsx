import { useLabStore } from "../../store/labStore";

const SPEEDS = [0.5, 1, 2, 4];

export function PlaybackControls() {
  const {
    isRunning,
    passDirection,
    currentStageIndex,
    currentBackwardStageIndex,
    stages,
    startPipeline,
    stepForward,
    stepBackward,
    skipToEnd,
    resetPipeline,
    pausePipeline,
    resumePipeline,
    speed,
    setSpeed,
    inputPixels,
    stepBackward_bwd,
    stepForward_bwd,
    skipToEnd_bwd,
    resetBackwardPass,
  } = useLabStore();

  const hasStarted = currentStageIndex >= 0;
  const atEnd = currentStageIndex >= stages.length - 1;
  const atStart = currentStageIndex < 0;

  const backwardStages = stages.filter((s) => !["input", "output"].includes(s.type));
  const atBackwardEnd = currentBackwardStageIndex >= backwardStages.length - 1;
  const atBackwardStart = currentBackwardStageIndex < 0;

  return (
    <div className="fixed bottom-[70px] left-0 right-0 z-30 px-4 md:bottom-4">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-700/70 bg-slate-950/85 px-3 py-2 backdrop-blur-xl">
        <div className="text-xs text-slate-400">
          {passDirection === "forward"
            ? `Stage ${Math.max(0, currentStageIndex + 1)} / ${stages.length}`
            : `Backward ${Math.max(0, currentBackwardStageIndex + 1)} / ${backwardStages.length}`}
        </div>

        <div className="flex items-center gap-2">
          {passDirection === "forward" ? (
            <>
              <button type="button" onClick={resetPipeline} disabled={!hasStarted} className="lab-btn">Reset</button>
              <button type="button" onClick={stepBackward} disabled={atStart} className="lab-btn">Prev</button>
              {!hasStarted ? (
                <button type="button" onClick={() => void startPipeline()} className="lab-btn lab-btn-primary">Start</button>
              ) : isRunning ? (
                <button type="button" onClick={pausePipeline} className="lab-btn lab-btn-warn">Pause</button>
              ) : (
                <button type="button" onClick={atEnd ? resetPipeline : resumePipeline} className="lab-btn lab-btn-primary">{atEnd ? "Restart" : "Resume"}</button>
              )}
              <button type="button" onClick={() => void stepForward()} disabled={atEnd} className="lab-btn">Next</button>
              <button type="button" onClick={() => void skipToEnd()} disabled={atEnd} className="lab-btn">Skip</button>
            </>
          ) : (
            <>
              <button type="button" onClick={resetBackwardPass} className="lab-btn">Back To Forward</button>
              <button type="button" onClick={stepForward_bwd} disabled={atBackwardStart} className="lab-btn">Prev</button>
              {isRunning ? (
                <button type="button" onClick={pausePipeline} className="lab-btn lab-btn-warn">Pause</button>
              ) : (
                <button type="button" onClick={() => void stepBackward_bwd()} disabled={atBackwardEnd} className="lab-btn lab-btn-primary">Run</button>
              )}
              <button type="button" onClick={() => void stepBackward_bwd()} disabled={atBackwardEnd} className="lab-btn">Next</button>
              <button type="button" onClick={() => void skipToEnd_bwd()} disabled={atBackwardEnd} className="lab-btn">Skip</button>
            </>
          )}
        </div>

        <div className="hidden items-center gap-1 sm:flex">
          <span className="text-xs text-slate-500">Speed</span>
          {SPEEDS.map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setSpeed(value)}
              className="rounded-md border px-2 py-1 text-xs"
              style={{
                borderColor: speed === value ? "rgba(34,211,238,0.6)" : "rgba(148,163,184,0.35)",
                background: speed === value ? "rgba(34,211,238,0.2)" : "rgba(15,23,42,0.5)",
                color: speed === value ? "#67e8f9" : "#94a3b8",
              }}
            >
              {value}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
