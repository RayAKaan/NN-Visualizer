import { useMemo } from "react";
import type { StageActivation, StageDefinition } from "../../../../types/pipeline";

interface Props {
  activation: StageActivation;
  stage: StageDefinition;
}

export function RecurrentCellViz({ activation, stage }: Props) {
  const gates = activation.gateValues;

  const avg = useMemo(() => {
    const mean = (arr: Float32Array) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
    if (!gates) return null;
    return {
      forget: mean(gates.forget),
      input: mean(gates.input),
      output: mean(gates.output),
    };
  }, [gates]);

  const hidden = Array.from(activation.outputData.slice(0, 96));

  return (
    <div className="space-y-3 text-xs">
      <div className="text-slate-400">{stage.name}: recurrent state update for this timestep.</div>
      {avg && (
        <div className="grid gap-2 sm:grid-cols-3">
          <div className="rounded-lg border border-rose-400/30 bg-rose-500/10 p-2 text-rose-200">Forget {Math.round(avg.forget * 100)}%</div>
          <div className="rounded-lg border border-emerald-400/30 bg-emerald-500/10 p-2 text-emerald-200">Input {Math.round(avg.input * 100)}%</div>
          <div className="rounded-lg border border-violet-400/30 bg-violet-500/10 p-2 text-violet-200">Output {Math.round(avg.output * 100)}%</div>
        </div>
      )}
      <div className="grid grid-cols-16 gap-1">
        {hidden.map((v, idx) => (
          <div key={idx} className="h-4 rounded" style={{ background: `rgba(168,85,247,${0.12 + Math.min(Math.abs(v), 1) * 0.88})` }} />
        ))}
      </div>
    </div>
  );
}
