import React from "react";
import { useComparisonStore, type Architecture } from "../../store/predictionStore";

const ARCHS: Architecture[] = ["ANN", "CNN", "RNN"];
const COLORS: Record<Architecture, string> = {
  ANN: "#f472b6",
  CNN: "#22d3ee",
  RNN: "#a855f7",
};

export function ProbabilityLandscape() {
  const viewMode = useComparisonStore((s) => s.viewMode);
  const toggleViewMode = useComparisonStore((s) => s.toggleViewMode);
  const results = useComparisonStore((s) => s.results);

  return (
    <div className="rounded-xl border border-cyan-400/10 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-semibold text-cyan-200">Probability Landscape</div>
        <div className="flex gap-1">
          {(["bars", "radial", "terrain"] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => toggleViewMode(mode)}
              className={`h-8 px-2 rounded border text-xs uppercase ${viewMode === mode ? "border-cyan-400/45 bg-cyan-500/15 text-cyan-200" : "border-white/10 bg-white/5 text-slate-300"}`}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      {viewMode === "bars" && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {ARCHS.map((arch) => {
            const probs = results[arch]?.probs ?? Array(10).fill(0);
            const best = probs.reduce((i, v, j, arr) => (v > arr[i] ? j : i), 0);
            return (
              <div key={arch} className="rounded-lg border border-white/10 bg-black/20 p-2">
                <div className="text-xs mb-1" style={{ color: COLORS[arch] }}>{arch}</div>
                <div className="h-24 grid grid-cols-10 gap-1 items-end">
                  {probs.map((p, i) => (
                    <div key={i} className="relative h-full rounded-sm bg-white/5 overflow-hidden">
                      <div
                        className="absolute bottom-0 left-0 right-0 origin-bottom transition-transform duration-300"
                        style={{
                          height: "100%",
                          transform: `scaleY(${Math.max(0.02, p)})`,
                          background: i === best ? COLORS[arch] : `${COLORS[arch]}99`,
                          boxShadow: i === best ? `0 0 14px ${COLORS[arch]}66` : "none",
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {viewMode === "radial" && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {ARCHS.map((arch) => {
            const probs = results[arch]?.probs ?? Array(10).fill(0);
            const best = probs.reduce((i, v, j, arr) => (v > arr[i] ? j : i), 0);
            return (
              <div key={arch} className="rounded-lg border border-white/10 bg-black/20 p-2 grid place-items-center">
                <svg width="190" height="130" viewBox="0 0 190 130">
                  <g transform="translate(95,65)">
                    {probs.map((p, i) => {
                      const a = (Math.PI * 2 * i) / 10 - Math.PI / 2;
                      const len = 18 + p * 40;
                      return <line key={i} x1={0} y1={0} x2={Math.cos(a) * len} y2={Math.sin(a) * len} stroke={i === best ? COLORS[arch] : `${COLORS[arch]}66`} strokeWidth={i === best ? 4 : 2} />;
                    })}
                    <circle r={14} fill="rgba(255,255,255,0.04)" stroke={`${COLORS[arch]}66`} />
                    <text x="0" y="1" fill={COLORS[arch]} fontSize="12" textAnchor="middle" dominantBaseline="middle">{results[arch]?.label ?? "-"}</text>
                  </g>
                </svg>
              </div>
            );
          })}
        </div>
      )}

      {viewMode === "terrain" && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {ARCHS.map((arch) => {
            const probs = results[arch]?.probs ?? Array(10).fill(0);
            const best = probs.reduce((i, v, j, arr) => (v > arr[i] ? j : i), 0);
            return (
              <div key={arch} className="rounded-lg border border-white/10 bg-black/20 p-2">
                <div className="h-[120px] flex items-end justify-center gap-1 [perspective:700px]">
                  {probs.map((p, i) => (
                    <div key={i} className="relative w-5">
                      <div
                        className="absolute bottom-0 w-5 rounded-t-sm transition-all duration-300"
                        style={{
                          height: Math.max(4, p * 90),
                          background: `linear-gradient(to top, ${COLORS[arch]}55, ${COLORS[arch]})`,
                          boxShadow: i === best ? `0 0 12px ${COLORS[arch]}66` : "none",
                          transform: "rotateX(20deg)",
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
