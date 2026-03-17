import type { StageActivation, StageDefinition } from "../../../types/pipeline";

interface ComparisonResult {
  stages: StageDefinition[];
  activations: Record<string, StageActivation>;
  prediction: { label: number | string; confidence: number; probs: number[] };
  totalTimeMs?: number;
}

interface Props {
  architecture: "ANN" | "CNN" | "RNN";
  accentColor: string;
  result: ComparisonResult | null;
  isLoading: boolean;
  scrollRef: (el: HTMLDivElement | null) => void;
}

export function ComparisonColumn({ architecture, accentColor, result, isLoading, scrollRef }: Props) {
  return (
    <section className="flex h-full flex-col border-r last:border-r-0" style={{ borderColor: "var(--glass-border)" }}>
      <header
        className="sticky top-0 z-10 border-b px-3 py-2 text-center"
        style={{
          borderColor: "var(--glass-border)",
          background: "var(--bg-panel)",
        }}
      >
        <div className="text-sm font-bold" style={{ color: accentColor }}>{architecture}</div>
        {result ? (
          <div className="mt-1 text-xs" style={{ color: "var(--text-3)" }}>
            {result.prediction.label} · {result.prediction.confidence.toFixed(1)}%
          </div>
        ) : null}
      </header>

      <div ref={scrollRef} className="comparison-scroll-container flex-1 space-y-2 overflow-y-auto p-2">
        {isLoading ? (
          <div className="py-10 text-center text-xs" style={{ color: "var(--text-4)" }}>Running {architecture}…</div>
        ) : null}

        {!isLoading && !result ? (
          <div className="py-10 text-center text-xs" style={{ color: "var(--text-4)" }}>No data</div>
        ) : null}

        {result
          ? result.stages.map((stage, i) => {
              const act = result.activations[stage.id];
              return (
                <article key={`${architecture}-${stage.id}`} className="rounded-lg border p-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-card)" }}>
                  <div className="mb-1 flex items-center gap-2">
                    <span className="grid h-5 w-5 place-items-center rounded text-[10px] font-bold" style={{ background: "var(--bg-panel)", color: accentColor }}>
                      {i + 1}
                    </span>
                    <span className="truncate text-xs" style={{ color: "var(--text-2)" }}>{stage.name}</span>
                  </div>
                  <div className="text-[10px] font-mono" style={{ color: "var(--text-4)" }}>
                    {stage.inputShape.join("x")} ? {stage.outputShape.join("x")}
                  </div>
                  {act ? (
                    <div className="mt-1 flex h-4 gap-px overflow-hidden rounded" style={{ background: "var(--bg-void)" }}>
                      {Array.from(act.outputData.slice(0, Math.min(24, act.outputData.length))).map((v, idx) => (
                        <div
                          key={idx}
                          className="flex-1"
                          style={{
                            background: accentColor,
                            opacity: Math.min(1, Math.abs(v) * 2 + 0.15),
                          }}
                        />
                      ))}
                    </div>
                  ) : null}
                </article>
              );
            })
          : null}
      </div>
    </section>
  );
}
