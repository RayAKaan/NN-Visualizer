import { useMemo } from "react";
import type { StageActivation } from "../../../../types/pipeline";

interface Props {
  activation: StageActivation;
}

export function ActivationViz({ activation }: Props) {
  const stats = useMemo(() => {
    const total = activation.inputData.length;
    let zeroed = 0;
    for (let i = 0; i < total; i += 1) if (activation.outputData[i] === 0) zeroed += 1;
    return { total, zeroed, passed: total - zeroed };
  }, [activation]);

  const input = Array.from(activation.inputData.slice(0, 32));
  const output = Array.from(activation.outputData.slice(0, 32));

  return (
    <div className="space-y-3">
      <div className="flex gap-2 text-xs">
        <div className="rounded-lg border border-rose-400/30 bg-rose-500/10 px-2 py-1 text-rose-200">Zeroed: {stats.zeroed}</div>
        <div className="rounded-lg border border-emerald-400/30 bg-emerald-500/10 px-2 py-1 text-emerald-200">Passed: {stats.passed}</div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        {input.map((v, i) => (
          <div key={i} className="rounded border border-slate-700 bg-slate-900/40 px-2 py-1">
            <span className="text-slate-400">{v.toFixed(3)}</span>
            <span className="mx-1 text-slate-500">?</span>
            <span className={output[i] === 0 ? "text-rose-300" : "text-emerald-300"}>{output[i].toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
