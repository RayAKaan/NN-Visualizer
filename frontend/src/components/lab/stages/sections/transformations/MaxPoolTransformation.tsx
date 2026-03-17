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
        <Card title="Before" value={`${inH}×${inW}`} />
        <Card title="Operation" value="max(2×2)" />
        <Card title="After" value={`${outH}×${outW}`} />
      </div>
      <p className="text-xs" style={{ color: "var(--text-3)" }}>
        Max pooling kept strongest local responses and reduced spatial detail by about {reduction.toFixed(1)}×.
      </p>
      <p className="text-[10px]" style={{ color: "var(--text-4)" }}>
        {JSON.stringify(stage.inputShape)} ? {JSON.stringify(stage.outputShape)}
      </p>
    </div>
  );
}

function Card({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-lg border p-3 text-center" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
      <div className="text-[10px] uppercase" style={{ color: "var(--text-4)" }}>{title}</div>
      <div className="text-sm font-mono" style={{ color: "var(--text-1)" }}>{value}</div>
    </div>
  );
}
