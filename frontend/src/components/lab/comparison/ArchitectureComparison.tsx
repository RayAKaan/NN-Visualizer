import { useRef } from "react";
import { useComparisonStore } from "../../../store/comparisonStore";
import { ComparisonColumn } from "./ComparisonColumn";
import { ComparisonInsight } from "./ComparisonInsight";
import { ComparisonSyncScroll } from "./ComparisonSyncScroll";

const ARCH_COLORS: Record<"ANN" | "CNN" | "RNN", string> = {
  ANN: "var(--arch-ann)",
  CNN: "var(--arch-cnn)",
  RNN: "var(--arch-rnn)",
};

export function ArchitectureComparison() {
  const isComparisonActive = useComparisonStore((s) => s.isComparisonActive);
  const loading = useComparisonStore((s) => s.loading);
  const results = useComparisonStore((s) => s.results);
  const stopComparison = useComparisonStore((s) => s.stopComparison);

  const scrollRefs = useRef<Record<"ANN" | "CNN" | "RNN", HTMLDivElement | null>>({
    ANN: null,
    CNN: null,
    RNN: null,
  });

  if (!isComparisonActive) return null;

  return (
    <section className="fixed inset-0 z-30 flex flex-col" style={{ background: "rgba(2,6,23,0.92)", backdropFilter: "blur(6px)" }}>
      <header className="flex items-center justify-between border-b px-4 py-2" style={{ borderColor: "var(--glass-border)", background: "var(--bg-card)" }}>
        <div className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>Architecture Comparison</div>
        <button
          type="button"
          onClick={stopComparison}
          className="rounded-md px-2 py-1 text-xs"
          style={{ border: "1px solid var(--glass-border)", color: "var(--text-3)", background: "var(--bg-panel)" }}
        >
          Close
        </button>
      </header>

      <ComparisonSyncScroll scrollRefs={scrollRefs}>
        <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-3">
          {(["ANN", "CNN", "RNN"] as const).map((arch) => (
            <ComparisonColumn
              key={arch}
              architecture={arch}
              accentColor={ARCH_COLORS[arch]}
              result={results[arch]}
              isLoading={loading[arch]}
              scrollRef={(el) => {
                scrollRefs.current[arch] = el;
              }}
            />
          ))}
        </div>
      </ComparisonSyncScroll>

      <ComparisonInsight results={results} />
    </section>
  );
}
