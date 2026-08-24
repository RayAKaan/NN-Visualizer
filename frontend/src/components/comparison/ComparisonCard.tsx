import React from "react";
import { useComparisonStore, type Architecture } from "../../store/predictionStore";
import { CNNFeatureGallery } from "../trace/CNNFeatureGallery";
import { RNNTemporalStrip } from "../trace/RNNTemporalStrip";

const ACCENT: Record<Architecture, string> = {
  ANN: "#0072B2",
  CNN: "#00806A",
  RNN: "#A64D85",
};

const confidenceColor = (c: number, accent: string) => {
  if (c >= 0.9) return "#15803D";
  if (c >= 0.7) return accent;
  if (c >= 0.5) return "#B45309";
  return "#B91C1C";
};

interface Props {
  arch: Architecture;
}

export function ComparisonCard({ arch }: Props) {
  const results = useComparisonStore((s) => s.results);
  const loading = useComparisonStore((s) => s.loading);
  const showTrace = useComparisonStore((s) => s.showTrace);
  const result = results[arch];
  const isLoading = loading[arch];

  const disagreement = useComparisonStore((s) => {
    const current = s.results[arch]?.label;
    if (current == null) return false;
    const others = (Object.keys(s.results) as Architecture[])
      .filter((a) => a !== arch)
      .map((a) => s.results[a]?.label)
      .filter((v): v is number => typeof v === "number");
    return others.some((v) => v !== current);
  });

  const accent = ACCENT[arch];

  return (
    <section className="rounded-xl border bg-white flex flex-col min-h-[320px]"
      style={{ borderColor: `${accent}33`, boxShadow: `0 8px 26px ${accent}12` }}>
      <header className="flex items-center justify-between px-3 py-2 border-b border-barley-line">
        <h3 className="text-sm font-semibold" style={{ color: accent }}>{arch}</h3>
        {disagreement && <span className="text-[12px] text-status-warning">Disagreement</span>}
      </header>

      <div className="flex-1 overflow-auto p-3 space-y-3">
        {isLoading ? (
          <div className="h-[170px] grid place-items-center text-xs text-ink-mute">
            <div className="inline-flex items-center gap-2"><span className="h-4 w-4 rounded-full border-2 border-ember-600/40 border-t-ember-500 animate-spin" />Running {arch}...</div>
          </div>
        ) : result ? (
          <div className="text-center">
            <div className="text-[90px] leading-none font-bold" style={{ color: accent }}>
              {result.label}
            </div>
            <div className="text-xs font-medium" style={{ color: confidenceColor(result.confidence, accent) }}>
              {(result.confidence * 100).toFixed(1)}%
            </div>
            <div className="text-[12px] text-ink-mute">Latency {result.latencyMs}ms</div>
            <div className="mt-2 h-10 flex items-end gap-1">
              {result.probs.map((p, i) => (
                <div key={i} className="flex-1 rounded-sm bg-barley-sunken overflow-hidden">
                  <div className="w-full rounded-sm" style={{ height: `${Math.max(2, p * 100)}%`, background: i === result.label ? accent : `${accent}88` }} />
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="h-[170px] grid place-items-center text-xs text-ink-faint">No prediction yet</div>
        )}

        {showTrace && result?.activations && (
          <div className="pt-2 border-t border-barley-line">
            {arch === "CNN" && <CNNFeatureGallery activations={result.activations} />}
            {arch === "RNN" && <RNNTemporalStrip activations={result.activations} />}
            {arch === "ANN" && (
              <div className="space-y-2">
                {(result.activations.layers ?? []).slice(0, 3).map((l: any) => (
                  <div key={l.id}>
                    <div className="text-[12px] text-ink-soft mb-1">{l.name}</div>
                    <div className="grid gap-[2px]" style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}>
                      {(Array.isArray(l.values) ? l.values : []).slice(0, 72).map((v: number, i: number) => (
                        <div key={i} className="h-4 rounded-[2px]" style={{ background: `rgba(194,65,12,${Math.max(0.1, Math.min(1, Math.abs(v))).toFixed(2)})` }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
