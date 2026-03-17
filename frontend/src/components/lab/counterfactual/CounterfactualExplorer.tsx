import { useMemo, useState } from "react";
import { useCounterfactualStore } from "../../../store/counterfactualStore";
import { useLabStore } from "../../../store/labStore";

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

export function CounterfactualExplorer() {
  const isOpen = useCounterfactualStore((s) => s.isOpen);
  const isBusy = useCounterfactualStore((s) => s.isBusy);
  const sensitivityMap = useCounterfactualStore((s) => s.sensitivityMap);
  const experiments = useCounterfactualStore((s) => s.experiments);
  const closeExplorer = useCounterfactualStore((s) => s.closeExplorer);
  const computeSensitivity = useCounterfactualStore((s) => s.computeSensitivity);
  const runCounterfactual = useCounterfactualStore((s) => s.runCounterfactual);
  const runMinimalFlip = useCounterfactualStore((s) => s.runMinimalFlip);

  const inputPixels = useLabStore((s) => s.inputPixels);
  const architecture = useLabStore((s) => s.architecture);
  const dataset = useLabStore((s) => s.dataset);

  const [delta, setDelta] = useState(0.1);

  const modified = useMemo(() => {
    const out = new Float32Array(inputPixels);
    const limit = Math.min(24, out.length);
    for (let i = 0; i < limit; i += 1) out[i] = clamp01(out[i] + delta);
    return out;
  }, [delta, inputPixels]);

  if (!isOpen) return null;

  return (
    <section
      className="fixed bottom-[78px] left-2 right-2 z-30 max-h-[56vh] overflow-y-auto rounded-2xl p-3 md:left-[88px]"
      style={{ background: "var(--bg-card)", border: "1px solid var(--glass-border)" }}
    >
      <header className="mb-2 flex items-center justify-between">
        <div className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>Counterfactual Explorer</div>
        <button type="button" onClick={closeExplorer} className="text-xs" style={{ color: "var(--text-3)" }}>Close</button>
      </header>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void computeSensitivity(inputPixels, architecture, dataset)}
          disabled={isBusy}
          className="rounded-lg px-2 py-1 text-xs"
          style={{ background: "var(--fwd-bg)", color: "var(--fwd)", border: "1px solid var(--fwd-border)" }}
        >
          Sensitivity Map
        </button>

        <label className="flex items-center gap-2 text-xs" style={{ color: "var(--text-3)" }}>
          Delta
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={delta}
            onChange={(e) => setDelta(Number(e.target.value))}
          />
          <span className="font-mono">{delta.toFixed(2)}</span>
        </label>

        <button
          type="button"
          onClick={() => void runCounterfactual(inputPixels, modified, architecture, dataset)}
          disabled={isBusy}
          className="rounded-lg px-2 py-1 text-xs"
          style={{ background: "var(--bg-panel)", color: "var(--text-2)", border: "1px solid var(--glass-border)" }}
        >
          Run Perturbation
        </button>

        <button
          type="button"
          onClick={() => void runMinimalFlip(inputPixels, architecture, dataset)}
          disabled={isBusy}
          className="rounded-lg px-2 py-1 text-xs"
          style={{ background: "var(--bg-panel)", color: "var(--warning)", border: "1px solid var(--glass-border)" }}
        >
          Minimal Flip
        </button>
      </div>

      {sensitivityMap ? (
        <div className="mt-3 rounded-lg p-2 text-xs" style={{ background: "var(--bg-panel)", color: "var(--text-3)", border: "1px solid var(--glass-border)" }}>
          Overall sensitivity: {sensitivityMap.overallSensitivity.toFixed(6)} · top pixels: {sensitivityMap.topSensitivePixels.length}
        </div>
      ) : null}

      {experiments.length > 0 ? (
        <div className="mt-3 space-y-1">
          {experiments.slice(0, 6).map((exp) => (
            <div key={exp.id} className="rounded-lg border p-2 text-xs" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
              <div style={{ color: "var(--text-2)" }}>
                Flip: {exp.predictionFlipped ? "yes" : "no"} · magnitude: {exp.perturbationMagnitude.toFixed(4)} · affected: {exp.affectedPixelCount}
              </div>
              <div style={{ color: "var(--text-4)" }}>
                {String(exp.originalPrediction.label)} ({exp.originalPrediction.confidence.toFixed(1)}%) ? {String(exp.modifiedPrediction.label)} ({exp.modifiedPrediction.confidence.toFixed(1)}%)
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
