import React, { useEffect, useMemo, useState } from "react";
import { useComparisonStore, type Architecture } from "../../store/predictionStore";

interface LayerNode {
  id: string;
  name: string;
  type: string;
}

const FALLBACK: Record<Architecture, LayerNode[]> = {
  ANN: [
    { id: "in", name: "Input", type: "input" },
    { id: "h1", name: "H1", type: "dense" },
    { id: "h2", name: "H2", type: "dense" },
    { id: "h3", name: "H3", type: "dense" },
    { id: "out", name: "Out", type: "dense" },
  ],
  CNN: [
    { id: "in", name: "Input", type: "input" },
    { id: "c1", name: "Conv1", type: "conv" },
    { id: "p1", name: "Pool", type: "pool" },
    { id: "c2", name: "Conv2", type: "conv" },
    { id: "out", name: "Out", type: "dense" },
  ],
  RNN: [
    { id: "in", name: "Input", type: "input" },
    { id: "lstm", name: "LSTM", type: "lstm" },
    { id: "dense", name: "Dense", type: "dense" },
    { id: "out", name: "Out", type: "dense" },
  ],
};

const colorForType = (type: string) => {
  if (type.includes("conv")) return "#22d3ee";
  if (type.includes("lstm") || type.includes("rnn")) return "#a855f7";
  if (type.includes("dense")) return "#f472b6";
  return "#00f0ff";
};

export function PipelineCascade() {
  const results = useComparisonStore((s) => s.results);
  const runId = useComparisonStore((s) => s.runId);
  const [active, setActive] = useState(-1);

  const layers = useMemo(() => {
    const archOrder: Architecture[] = ["ANN", "CNN", "RNN"];
    for (const arch of archOrder) {
      const maybe = results[arch]?.activations?.layers;
      if (Array.isArray(maybe) && maybe.length > 0) {
        return maybe.map((l: any, i: number) => ({
          id: String(l.id ?? `${arch}-${i}`),
          name: String(l.name ?? `L${i + 1}`),
          type: String(l.type ?? "dense"),
        }));
      }
    }
    return FALLBACK.ANN;
  }, [results]);

  useEffect(() => {
    if (runId === 0 || layers.length === 0) return;
    setActive(0);
    const timers: number[] = [];
    for (let i = 0; i < layers.length; i += 1) {
      timers.push(window.setTimeout(() => setActive(i), i * 120));
    }
    timers.push(window.setTimeout(() => setActive(-1), layers.length * 120 + 450));
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [layers, runId]);

  return (
    <div className="rounded-xl border border-cyan-400/10 bg-slate-900/60 p-3">
      <div className="text-xs text-slate-400 mb-2">Layer Cascade</div>
      <div className="flex items-center gap-2 overflow-x-auto pb-1">
        {layers.map((layer, i) => (
          <React.Fragment key={layer.id}>
            <div
              className="shrink-0 w-12 h-12 rounded-md border grid place-items-center text-[10px] font-mono transition-all duration-200"
              style={{
                borderColor: i === active ? `${colorForType(layer.type)}aa` : `${colorForType(layer.type)}44`,
                background: i === active ? `${colorForType(layer.type)}22` : "rgba(255,255,255,0.03)",
                opacity: i === active ? 1 : 0.65,
                transform: i === active ? "scale(1.05)" : "scale(0.95)",
                boxShadow: i === active ? `0 0 18px ${colorForType(layer.type)}66` : "none",
              }}
              aria-label={`Layer ${layer.name}`}
            >
              {layer.name}
            </div>
            {i < layers.length - 1 && (
              <div className="relative shrink-0 w-7 h-[2px] bg-white/15 overflow-hidden rounded">
                <span
                  className="absolute top-1/2 -translate-y-1/2 h-1.5 w-1.5 rounded-full"
                  style={{
                    background: "#67e8f9",
                    animation: active >= i ? "ann-flow-dot 0.7s linear infinite" : "none",
                  }}
                />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
