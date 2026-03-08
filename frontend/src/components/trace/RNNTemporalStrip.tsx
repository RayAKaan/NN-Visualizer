import React, { useMemo, useState } from "react";

interface Step {
  hidden: number[];
  gates?: { f?: number[]; i?: number[]; o?: number[] };
  attention?: number;
}

interface Props {
  activations: {
    timesteps?: Step[];
  };
}

const avg = (arr?: number[]) => {
  if (!arr || arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
};

export function RNNTemporalStrip({ activations }: Props) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const steps = useMemo(() => (Array.isArray(activations.timesteps) ? activations.timesteps : []), [activations.timesteps]);
  if (steps.length === 0) return <div className="text-xs text-slate-400">No RNN temporal data available.</div>;

  const maxMag = Math.max(0.001, ...steps.map((s) => Math.abs(avg(s.hidden))));
  const accent = "168,85,247";

  return (
    <div className="space-y-2" style={{ transform: "perspective(900px) rotateX(8deg)" }}>
      <div className="text-xs font-semibold text-violet-200">Temporal Hidden-State Evolution</div>
      <div className="flex overflow-x-auto gap-1 py-2">
        {steps.map((step, idx) => {
          const mag = Math.abs(avg(step.hidden)) / maxMag;
          const height = Math.round(mag * 92);
          const f = avg(step.gates?.f);
          const i = avg(step.gates?.i);
          const o = avg(step.gates?.o);
          return (
            <div
              key={idx}
              className="w-5 shrink-0 flex flex-col items-center"
              onMouseEnter={() => setHoverIdx(idx)}
              onMouseLeave={() => setHoverIdx(null)}
            >
              <div className="flex gap-[1px] mb-1 w-full">
                <div className="h-1 bg-rose-400 rounded-sm" style={{ width: `${Math.round(f * 100)}%` }} />
                <div className="h-1 bg-emerald-400 rounded-sm" style={{ width: `${Math.round(i * 100)}%` }} />
                <div className="h-1 bg-violet-400 rounded-sm" style={{ width: `${Math.round(o * 100)}%` }} />
              </div>
              <div
                className="w-full rounded-t-sm transition-transform duration-150"
                style={{
                  height: `${Math.max(5, height)}px`,
                  backgroundColor: `rgba(${accent},${Math.max(0.18, mag).toFixed(2)})`,
                  boxShadow: hoverIdx === idx ? "0 0 10px rgba(168,85,247,0.6)" : "none",
                  transform: hoverIdx === idx ? "translateY(-3px) scale(1.1)" : "none",
                }}
              />
              <span className="text-[10px] text-slate-400 mt-0.5">{idx}</span>
            </div>
          );
        })}
      </div>
      {hoverIdx !== null && (
        <div className="rounded-md border border-white/10 bg-black/30 p-2">
          <div className="text-[11px] text-slate-300 mb-1">Timestep {hoverIdx} snapshot</div>
          <div className="grid grid-cols-16 gap-[2px]">
            {(steps[hoverIdx].hidden ?? []).slice(0, 64).map((v, i) => (
              <div key={i} className="h-3 rounded-[2px]" style={{ backgroundColor: `rgba(${accent},${Math.max(0.1, Math.min(1, Math.abs(v))).toFixed(2)})` }} />
            ))}
          </div>
          <div className="text-[10px] text-slate-400 mt-1">Attention: {Number(steps[hoverIdx].attention ?? 0).toFixed(3)}</div>
        </div>
      )}
    </div>
  );
}
