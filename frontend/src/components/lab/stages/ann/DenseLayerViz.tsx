import type { StageActivation, StageDefinition } from "../../../../types/pipeline";

interface Props {
  activation: StageActivation;
  stage: StageDefinition;
}

export function DenseLayerViz({ activation, stage }: Props) {
  const input = Array.from(activation.inputData.slice(0, 48));
  const output = Array.from(activation.outputData.slice(0, 32));
  const maxIn = Math.max(...input.map((v) => Math.abs(v)), 0.01);
  const maxOut = Math.max(...output.map((v) => Math.abs(v)), 0.01);

  return (
    <div className="space-y-3">
      <div className="text-xs text-ink-mute">{stage.name}: weighted transform from input vector to output units.</div>
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <div className="mb-1 text-[12px] text-ink-faint">Input sample</div>
          <div className="space-y-1">
            {input.map((v, i) => (
              <div key={i} className="h-2 rounded bg-barley-wash">
                <div className="h-full rounded" style={{ width: `${(Math.abs(v) / maxIn) * 100}%`, background: v >= 0 ? "rgba(194,65,12,0.8)" : "rgba(0,114,178,0.8)" }} />
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="mb-1 text-[12px] text-ink-faint">Output sample</div>
          <div className="space-y-1">
            {output.map((v, i) => (
              <div key={i} className="h-2 rounded bg-barley-wash">
                <div className="h-full rounded bg-status-success/12" style={{ width: `${(Math.abs(v) / maxOut) * 100}%` }} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
