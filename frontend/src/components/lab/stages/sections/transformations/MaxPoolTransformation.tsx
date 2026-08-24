import type { ReactNode } from "react";
import type { StageActivation, StageDefinition } from "../../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function MaxPoolTransformation({ stage, activation }: Props) {
  const inShape = activation.metadata.inputShape;
  const outShape = activation.metadata.outputShape;
  const inH = inShape[inShape.length - 2] ?? 0;
  const inW = inShape[inShape.length - 1] ?? 0;
  const outH = outShape[outShape.length - 2] ?? 0;
  const outW = outShape[outShape.length - 1] ?? 0;
  const reduction = (inH * inW) / Math.max(1, outH * outW);

  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-3">
        <Card title="Before" value={<>{inH}&times;{inW}</>} />
        <Card title="Operation" value={<>max(2&times;2)</>} />
        <Card title="After" value={<>{outH}&times;{outW}</>} />
      </div>
      <p className="text-xs text-ink-mute">
        Max pooling kept strongest local responses and reduced spatial detail by about {reduction.toFixed(1)}&times;.
      </p>
      <p className="text-[12px] text-ink-faint">
        {JSON.stringify(stage.inputShape)} &rarr; {JSON.stringify(stage.outputShape)}
      </p>
    </div>
  );
}

function Card({ title, value }: { title: string; value: ReactNode }) {
  return (
    <div className="rounded-lg border border-barley-linestrong bg-white p-3 text-center">
      <div className="text-[12px] uppercase text-ink-faint">{title}</div>
      <div className="text-sm font-mono text-ink">{value}</div>
    </div>
  );
}
