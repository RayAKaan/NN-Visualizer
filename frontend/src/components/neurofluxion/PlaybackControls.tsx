import React from "react";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useNeurofluxStore } from "../../store/useNeurofluxStore";

interface Props {
  connectionStatus: "connecting" | "connected" | "disconnected";
}

export default function PlaybackControls({ connectionStatus }: Props) {
  const playbackState = useNeurofluxStore((s) => s.playbackState);
  const currentEpoch = useNeurofluxStore((s) => s.currentEpoch);
  const currentArchitecture = useNeurofluxStore((s) => s.currentArchitecture);
  const history = useNeurofluxStore((s) => s.history);
  const play = useNeurofluxStore((s) => s.play);
  const pause = useNeurofluxStore((s) => s.pause);
  const stepForward = useNeurofluxStore((s) => s.stepForward);
  const stepBackward = useNeurofluxStore((s) => s.stepBackward);
  const scrubToEpoch = useNeurofluxStore((s) => s.scrubToEpoch);

  const isPaused = playbackState === "paused";
  const canScrub = currentArchitecture === "ann";
  const maxEpoch = Math.max(0, history.length - 1);
  const statusDot =
    connectionStatus === "connected"
      ? "bg-emerald-400"
      : connectionStatus === "connecting"
        ? "bg-amber-400"
        : "bg-rose-400";

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-2 flex items-center gap-3">
      <div className={`h-2.5 w-2.5 rounded-full ${statusDot}`} />
      <div className="text-xs text-slate-400 min-w-[85px]">{connectionStatus}</div>

      <button
        onClick={() => stepBackward()}
        disabled={!isPaused || !canScrub || currentEpoch <= 0}
        className="p-1.5 rounded border border-slate-600 bg-slate-800 disabled:opacity-40"
        title="Step Back"
      >
        <SkipBack size={14} />
      </button>

      {isPaused ? (
        <button
          onClick={() => play()}
          className="p-1.5 rounded border border-cyan-600 bg-cyan-700/70"
          title="Play"
        >
          <Play size={14} />
        </button>
      ) : (
        <button
          onClick={() => pause()}
          className="p-1.5 rounded border border-fuchsia-600 bg-fuchsia-700/70"
          title="Pause"
        >
          <Pause size={14} />
        </button>
      )}

      <button
        onClick={() => stepForward()}
        disabled={!isPaused || !canScrub || currentEpoch >= maxEpoch}
        className="p-1.5 rounded border border-slate-600 bg-slate-800 disabled:opacity-40"
        title="Step Forward"
      >
        <SkipForward size={14} />
      </button>

      <input
        type="range"
        min={0}
        max={maxEpoch}
        value={Math.min(currentEpoch, maxEpoch)}
        onChange={(e) => scrubToEpoch(parseInt(e.target.value, 10))}
        disabled={!canScrub}
        className="flex-1 accent-cyan-400"
      />
      <div className="text-xs font-mono text-slate-300 min-w-[72px] text-right">
        epoch {Math.min(currentEpoch, maxEpoch)} / {maxEpoch}
      </div>
    </div>
  );
}
