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
        <div className="rounded-lg border border-arch-rnn/40 bg-arch-rnn/12 px-2 py-1 text-arch-rnn">Zeroed: {stats.zeroed}</div>
        <div className="rounded-lg border border-status-success/40 bg-status-success/12 px-2 py-1 text-status-success">Passed: {stats.passed}</div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        {input.map((v, i) => (
          <div key={i} className="rounded border border-barley-linestrong bg-white px-2 py-1">
            <span className="text-ink-mute">{v.toFixed(3)}</span>
            <span className="mx-1 text-ink-faint">?</span>
            <span className={output[i] === 0 ? "text-arch-rnn" : "text-status-success"}>{output[i].toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
