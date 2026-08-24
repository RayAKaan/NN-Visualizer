import { useMemo, useState } from "react";
import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
  highlightedVariable: string | null;
  onHighlightVariable: (v: string | null) => void;
}

export function DenseTransformation({ stage, activation, highlightedVariable, onHighlightVariable }: Props) {
  const inputSize = Number(stage.params?.inputDim ?? stage.inputShape[stage.inputShape.length - 1] ?? activation.inputData.length);
  const outputSize = Number(stage.params?.units ?? stage.outputShape[stage.outputShape.length - 1] ?? activation.outputData.length);
  const inSample = Array.from(activation.inputData.slice(0, Math.min(24, activation.inputData.length)));
  const outSample = Array.from(activation.outputData.slice(0, Math.min(24, activation.outputData.length)));
  const maxIn = Math.max(...inSample.map((v) => Math.abs(v)), 0.01);
  const maxOut = Math.max(...outSample.map((v) => Math.abs(v)), 0.01);
  const [selected, setSelected] = useState<number | null>(null);

  const topContrib = useMemo(() => {
    if (selected == null || !activation.weights) return [] as Array<{ i: number; x: number; w: number; p: number }>;
    const list: Array<{ i: number; x: number; w: number; p: number }> = [];
    const limit = Math.min(inputSize, activation.inputData.length, 64);
    for (let i = 0; i < limit; i += 1) {
      const w = activation.weights[selected * inputSize + i] ?? 0;
      const x = activation.inputData[i] ?? 0;
      list.push({ i, x, w, p: x * w });
    }
    return list.sort((a, b) => Math.abs(b.p) - Math.abs(a.p)).slice(0, 8);
  }, [activation.inputData, activation.weights, inputSize, selected]);

  const labelColor = (v: string) => (highlightedVariable === v ? "text-arch-ann" : "text-ink-faint");

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-center gap-3">
        <div onMouseEnter={() => onHighlightVariable("x")} onMouseLeave={() => onHighlightVariable(null)}>
          <div className={`mb-1 text-[12px] ${labelColor("x")}`}>Input x [{inputSize}]</div>
          <div className="space-y-px">
            {inSample.map((v, i) => (
              <div key={i} className="h-2.5 w-20 rounded bg-white">
                <div className="h-full rounded bg-ember-600" style={{ width: `${(Math.abs(v) / maxIn) * 100}%` }} />
              </div>
            ))}
          </div>
        </div>
        <div className="pt-8 text-xl text-ink-faint">&times;</div>
        <div onMouseEnter={() => onHighlightVariable("W")} onMouseLeave={() => onHighlightVariable(null)}>
          <div className={`mb-1 text-[12px] ${labelColor("W")}`}>
            Weights W [{outputSize}
            {"\u00d7"}
            {inputSize}]
          </div>
          <div className="grid grid-cols-8 gap-px rounded border border-barley-linestrong bg-barley-page p-1">
            {Array.from(activation.weights?.slice(0, 64) ?? []).map((w, i) => (
              <div key={i} className="h-2.5 w-2.5 rounded-sm" style={{ background: w >= 0 ? `rgba(194,65,12,${Math.min(Math.abs(w) * 3, 1)})` : `rgba(0,114,178,${Math.min(Math.abs(w) * 3, 1)})` }} />
            ))}
          </div>
        </div>
        <div className="pt-8 text-xl text-ink-faint">=</div>
        <div onMouseEnter={() => onHighlightVariable("z")} onMouseLeave={() => onHighlightVariable(null)}>
          <div className={`mb-1 text-[12px] ${labelColor("z")}`}>Output z [{outputSize}]</div>
          <div className="space-y-px">
            {outSample.map((v, i) => (
              <button
                key={i}
                type="button"
                className={`h-2.5 w-20 rounded bg-white ${selected === i ? "ring-2 ring-status-warning" : ""}`}
                onClick={() => setSelected(selected === i ? null : i)}
                aria-label={`Select output neuron ${i}`}
              >
                <div className="h-full rounded bg-ember-600" style={{ width: `${(Math.abs(v) / maxOut) * 100}%` }} />
              </button>
            ))}
          </div>
        </div>
      </div>
      {selected != null ? (
        <div className="rounded-lg border border-barley-linestrong bg-white p-3 text-xs">
          <div className="mb-2 font-semibold text-status-warning">Neuron z[{selected}] contribution trace</div>
          <div className="space-y-1 font-mono text-ink-soft">
            {topContrib.map((c) => (
              <div key={c.i} className="flex items-center gap-2">
                <span className="text-ink-faint">x[{c.i}]</span>
                <span>{c.x.toFixed(3)}</span>
                <span className="text-ink-faint">&times;</span>
                <span>{c.w.toFixed(3)}</span>
                <span className="text-ink-faint">=</span>
                <span className={c.p >= 0 ? "text-ember-800" : "text-arch-ann"}>{c.p.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
