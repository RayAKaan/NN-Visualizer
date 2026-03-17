import type { StageActivation } from "../../../types/pipeline";

interface ArchResult {
  prediction: { label: number | string; confidence: number; probs: number[] };
  activations: Record<string, StageActivation>;
  totalTimeMs?: number;
}

interface Props {
  results: Record<"ANN" | "CNN" | "RNN", ArchResult | null>;
}

export function ComparisonInsight({ results }: Props) {
  if (!results.ANN || !results.CNN || !results.RNN) return null;

  const tuples: Array<["ANN" | "CNN" | "RNN", ArchResult]> = [
    ["ANN", results.ANN],
    ["CNN", results.CNN],
    ["RNN", results.RNN],
  ];

  const best = tuples.sort((a, b) => b[1].prediction.confidence - a[1].prediction.confidence)[0];
  const labelSet = new Set([results.ANN.prediction.label, results.CNN.prediction.label, results.RNN.prediction.label]);

  return (
    <div className="border-t px-3 py-2 text-xs" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)", color: "var(--text-2)" }}>
      {labelSet.size === 1
        ? `All three agree on ${String(best[1].prediction.label)}. ${best[0]} is most confident (${best[1].prediction.confidence.toFixed(1)}%).`
        : `Architectures disagree. ${best[0]} is most confident (${best[1].prediction.confidence.toFixed(1)}%) on ${String(best[1].prediction.label)}.`}
    </div>
  );
}
