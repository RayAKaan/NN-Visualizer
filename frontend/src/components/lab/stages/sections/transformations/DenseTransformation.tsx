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

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-center gap-3">
        <div onMouseEnter={() => onHighlightVariable("x")} onMouseLeave={() => onHighlightVariable(null)}>
          <div className="mb-1 text-[10px]" style={{ color: highlightedVariable === "x" ? "var(--fwd)" : "var(--text-4)" }}>Input x [{inputSize}]</div>
          <div className="space-y-px">
            {inSample.map((v, i) => (
              <div key={i} className="h-2.5 w-20 rounded" style={{ background: `linear-gradient(to right, var(--activation-mid) ${(Math.abs(v) / maxIn) * 100}%, var(--bg-panel) 0%)` }} />
            ))}
          </div>
        </div>
        <div className="pt-8 text-xl" style={{ color: "var(--text-4)" }}>×</div>
        <div onMouseEnter={() => onHighlightVariable("W")} onMouseLeave={() => onHighlightVariable(null)}>
          <div className="mb-1 text-[10px]" style={{ color: highlightedVariable === "W" ? "var(--fwd)" : "var(--text-4)" }}>Weights W [{outputSize}×{inputSize}]</div>
          <div className="grid grid-cols-8 gap-px rounded border p-1" style={{ borderColor: "var(--glass-border)", background: "var(--bg-void)" }}>
            {Array.from(activation.weights?.slice(0, 64) ?? []).map((w, i) => (
              <div key={i} className="h-2.5 w-2.5 rounded-sm" style={{ background: w >= 0 ? `rgba(94,234,212,${Math.min(Math.abs(w) * 3, 1)})` : `rgba(232,121,160,${Math.min(Math.abs(w) * 3, 1)})` }} />
            ))}
          </div>
        </div>
        <div className="pt-8 text-xl" style={{ color: "var(--text-4)" }}>=</div>
        <div onMouseEnter={() => onHighlightVariable("z")} onMouseLeave={() => onHighlightVariable(null)}>
          <div className="mb-1 text-[10px]" style={{ color: highlightedVariable === "z" ? "var(--fwd)" : "var(--text-4)" }}>Output z [{outputSize}]</div>
          <div className="space-y-px">
            {outSample.map((v, i) => (
              <button key={i} type="button" className="h-2.5 w-20 rounded text-left" onClick={() => setSelected(selected === i ? null : i)} style={{ border: selected === i ? "1px solid var(--warning)" : "none", background: `linear-gradient(to right, var(--fwd) ${(Math.abs(v) / maxOut) * 100}%, var(--bg-panel) 0%)` }} />
            ))}
          </div>
        </div>
      </div>
      {selected != null ? (
        <div className="rounded-lg border p-3 text-xs" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
          <div className="mb-2 font-semibold" style={{ color: "var(--warning)" }}>Neuron z[{selected}] contribution trace</div>
          <div className="space-y-1 font-mono" style={{ color: "var(--text-2)" }}>
            {topContrib.map((c) => (
              <div key={c.i} className="flex items-center gap-2">
                <span style={{ color: "var(--text-4)" }}>x[{c.i}]</span>
                <span>{c.x.toFixed(3)}</span>
                <span style={{ color: "var(--text-4)" }}>×</span>
                <span>{c.w.toFixed(3)}</span>
                <span style={{ color: "var(--text-4)" }}>=</span>
                <span style={{ color: c.p >= 0 ? "var(--weight-positive-strong)" : "var(--weight-negative-strong)" }}>{c.p.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
