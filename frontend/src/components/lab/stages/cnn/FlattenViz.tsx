import type { StageActivation } from "../../../../types/pipeline";

interface Props {
  activation: StageActivation;
}

export function FlattenViz({ activation }: Props) {
  return (
    <div className="space-y-2 text-xs">
      <div className="text-ink-mute">Flatten converts multi-dimensional maps into a single feature vector.</div>
      <div className="rounded-lg border border-barley-linestrong bg-white p-2 font-mono text-ink-soft">
        {JSON.stringify(activation.metadata.inputShape)} &rarr; [{activation.outputData.length}]
      </div>
      <div className="grid grid-cols-8 gap-1">
        {Array.from(activation.outputData.slice(0, 64)).map((v, idx) => (
          <div key={idx} className="h-3 rounded" style={{ background: `rgba(194,65,12,${0.15 + Math.min(Math.abs(v), 1) * 0.85})` }} />
        ))}
      </div>
    </div>
  );
}
