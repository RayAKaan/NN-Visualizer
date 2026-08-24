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
      className="fixed bottom-[192px] left-2 right-2 z-30 max-h-[56vh] overflow-y-auto rounded-2xl border border-ink/10 bg-white p-3 shadow-pop backdrop-blur-xl md:left-[88px] md:bottom-[140px]"
    >
      <header className="mb-2 flex items-center justify-between">
        <div className="text-sm font-semibold text-ink">Counterfactual Explorer</div>
        <button type="button" onClick={closeExplorer} className="text-xs text-ink-mute hover:text-ember-700">Close</button>
      </header>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void computeSensitivity(inputPixels, architecture, dataset)}
          disabled={isBusy}
          className="rounded-lg border border-arch-ann/25 bg-arch-ann/10 px-2 py-1 text-xs text-arch-ann"
        >
          Sensitivity Map
        </button>

        <label className="flex items-center gap-2 text-xs text-ink-mute">
          Delta
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={delta}
            onChange={(e) => setDelta(Number(e.target.value))}
            className="neural-slider w-28"
          />
          <span className="font-mono">{delta.toFixed(2)}</span>
        </label>

        <button
          type="button"
          onClick={() => void runCounterfactual(inputPixels, modified, architecture, dataset)}
          disabled={isBusy}
          className="rounded-lg border border-barley-linestrong bg-barley-page px-2 py-1 text-xs text-ink-soft hover:border-ember-600/40"
        >
          Run Perturbation
        </button>

        <button
          type="button"
          onClick={() => void runMinimalFlip(inputPixels, architecture, dataset)}
          disabled={isBusy}
          className="rounded-lg border border-status-warning/35 bg-status-warning/10 px-2 py-1 text-xs text-status-warning"
        >
          Minimal Flip
        </button>
      </div>

      {sensitivityMap ? (
        <div className="mt-3 rounded-lg border border-barley-linestrong bg-barley-page p-2 text-xs text-ink-mute">
          Overall sensitivity: {sensitivityMap.overallSensitivity.toFixed(6)} {"\u00b7"} top pixels: {sensitivityMap.topSensitivePixels.length}
        </div>
      ) : null}

      {experiments.length > 0 ? (
        <div className="mt-3 space-y-1">
          {experiments.slice(0, 6).map((exp) => (
            <div key={exp.id} className="rounded-lg border border-barley-linestrong bg-barley-page p-2 text-xs">
              <div className="text-ink-soft">
                Flip: {exp.predictionFlipped ? "yes" : "no"} {"\u00b7"} magnitude: {exp.perturbationMagnitude.toFixed(4)} {"\u00b7"} affected: {exp.affectedPixelCount}
              </div>
              <div className="text-ink-faint">
                {String(exp.originalPrediction.label)} ({exp.originalPrediction.confidence.toFixed(1)}%) &rarr; {String(exp.modifiedPrediction.label)} ({exp.modifiedPrediction.confidence.toFixed(1)}%)
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
