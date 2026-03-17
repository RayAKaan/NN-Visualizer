import type { StageActivation } from "../../../../types/pipeline";

interface Props {
  activation: StageActivation;
}

export function PoolingViz({ activation }: Props) {
  return (
    <div className="space-y-2 text-xs">
      <div className="text-slate-400">Pooling compresses the feature map while preserving strongest activations.</div>
      <div className="grid gap-2 sm:grid-cols-2">
        <div className="rounded-lg border border-slate-700 bg-slate-900/60 p-2">
          <div className="text-slate-500">Input elements</div>
          <div className="font-mono text-slate-200">{activation.inputData.length}</div>
        </div>
        <div className="rounded-lg border border-slate-700 bg-slate-900/60 p-2">
          <div className="text-slate-500">Output elements</div>
          <div className="font-mono text-slate-200">{activation.outputData.length}</div>
        </div>
      </div>
    </div>
  );
}
