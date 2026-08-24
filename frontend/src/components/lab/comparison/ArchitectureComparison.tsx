import { useRef } from "react";
import { useComparisonStore } from "../../../store/comparisonStore";
import { ComparisonColumn } from "./ComparisonColumn";
import { ComparisonInsight } from "./ComparisonInsight";
import { ComparisonSyncScroll } from "./ComparisonSyncScroll";

const ARCH_COLORS: Record<"ANN" | "CNN" | "RNN", string> = {
  ANN: "#0072B2",
  CNN: "#00806A",
  RNN: "#A64D85",
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
    <section className="fixed inset-0 z-30 flex flex-col bg-barley-page/95 backdrop-blur-sm">
      <header className="flex items-center justify-between border-b border-barley-linestrong bg-white px-4 py-2">
        <div className="text-sm font-semibold text-ink">Architecture Comparison</div>
        <button
          type="button"
          onClick={stopComparison}
          className="rounded-md border border-barley-linestrong bg-barley-page px-2 py-1 text-xs text-ink-mute hover:border-ember-600/40"
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
