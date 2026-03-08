import React, { useEffect, useMemo, useState } from "react";
import { useComparisonStore, type Architecture } from "../../store/predictionStore";

const ARCHS: Architecture[] = ["ANN", "CNN", "RNN"];

export function DisagreementHighlighter() {
  const results = useComparisonStore((s) => s.results);
  const [visible, setVisible] = useState(false);

  const message = useMemo(() => {
    const labels = ARCHS
      .map((a) => ({ arch: a, label: results[a]?.label }))
      .filter((x): x is { arch: Architecture; label: number } => typeof x.label === "number");
    if (labels.length < 2) return null;
    const unique = new Set(labels.map((x) => x.label));
    if (unique.size <= 1) return null;
    return labels.map((x) => `${x.arch}:${x.label}`).join(", ");
  }, [results]);

  useEffect(() => {
    if (!message) return;
    setVisible(true);
    const t = window.setTimeout(() => setVisible(false), 3000);
    return () => window.clearTimeout(t);
  }, [message]);

  if (!message || !visible) return null;

  return (
    <div className="fixed inset-0 pointer-events-none z-40 flex items-start justify-center pt-16" aria-live="assertive">
      <div className="rounded-full border border-amber-400/40 bg-amber-500/15 px-4 py-2 text-xs text-amber-200 shadow-[0_0_20px_rgba(245,158,11,0.35)] animate-pulse">
        Models disagree: {message}
      </div>
    </div>
  );
}
