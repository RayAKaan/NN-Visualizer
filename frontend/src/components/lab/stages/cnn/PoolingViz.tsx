import type { StageActivation } from "../../../../types/pipeline";

interface Props {
  activation: StageActivation;
}

export function PoolingViz({ activation }: Props) {
  return (
    <div className="space-y-2 text-xs">
      <div className="text-ink-mute">Pooling compresses the feature map while preserving strongest activations.</div>
      <div className="grid gap-2 sm:grid-cols-2">
        <div className="rounded-lg border border-barley-linestrong bg-white p-2">
          <div className="text-ink-faint">Input elements</div>
          <div className="font-mono text-ink">{activation.inputData.length}</div>
        </div>
        <div className="rounded-lg border border-barley-linestrong bg-white p-2">
          <div className="text-ink-faint">Output elements</div>
          <div className="font-mono text-ink">{activation.outputData.length}</div>
        </div>
      </div>
    </div>
  );
}
