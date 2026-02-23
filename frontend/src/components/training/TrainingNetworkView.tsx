import React, { useMemo } from "react";
import NeuronGrid from "../prediction/NeuronGrid";
import { LayerActivation, GradientInfo } from "../../types";

interface Props {
  activations: Record<string, LayerActivation> | null;
  gradients: Record<string, GradientInfo> | null;
  modelType: string;
}

export default function TrainingNetworkView({ activations, gradients }: Props) {
  const layers = useMemo(() => {
    if (!activations) return [];
    const result: { name: string; values: number[]; gradNorm: number }[] = [];
    for (const [name, act] of Object.entries(activations)) {
      if (act.type === "dense" && act.values) {
        const gradInfo = gradients?.[`${name}/kernel:0`] || gradients?.[name];
        result.push({ name, values: act.values, gradNorm: gradInfo?.norm ?? 0 });
      }
    }
    return result;
  }, [activations, gradients]);

  const totalGradNorm = (gradients as any)?.total_norm ?? 0;

  if (layers.length === 0) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "var(--text-muted)",
          fontSize: 13,
        }}
      >
        {activations ? "Processing..." : "Start training to see network activity"}
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        padding: 16,
        gap: 12,
      }}
    >
      <div
        style={{
          fontSize: 11,
          fontFamily: "var(--font-mono)",
          color: "var(--text-muted)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span>Gradient Norm:</span>
        <span
          style={{
            color:
              totalGradNorm > 5
                ? "var(--accent-red)"
                : totalGradNorm > 1
                ? "var(--accent-amber)"
                : "var(--accent-green)",
            fontWeight: 600,
          }}
        >
          {totalGradNorm.toFixed(4)}
        </span>
        <div
          style={{
            width: 60,
            height: 4,
            background: "var(--bg-secondary)",
            borderRadius: 2,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${Math.min(totalGradNorm * 10, 100)}%`,
              background:
                totalGradNorm > 5
                  ? "var(--accent-red)"
                  : totalGradNorm > 1
                  ? "var(--accent-amber)"
                  : "var(--accent-green)",
              borderRadius: 2,
              transition: "width 100ms",
            }}
          />
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
        {layers.map((layer, idx) => (
          <React.Fragment key={layer.name}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <NeuronGrid
                activations={layer.values}
                columns={
                  layer.values.length <= 10
                    ? layer.values.length
                    : Math.ceil(Math.sqrt(layer.values.length))
                }
                label=""
              />
              <span style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                {layer.name} ({layer.values.length})
              </span>
              <span
                style={{
                  fontSize: 8,
                  fontFamily: "var(--font-mono)",
                  color: layer.gradNorm > 1 ? "var(--accent-amber)" : "var(--accent-green)",
                }}
              >
                ∇ {layer.gradNorm.toFixed(3)}
              </span>
            </div>

            {idx < layers.length - 1 && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                <span style={{ fontSize: 14, color: "var(--accent-cyan)", opacity: 0.6 }}>→</span>
                <span
                  className="gradient-flow-line"
                  style={{
                    fontSize: 12,
                    color: "var(--accent-red)",
                    opacity: Math.min(0.3 + totalGradNorm * 0.2, 1),
                  }}
                >
                  ←
                </span>
                <span style={{ fontSize: 8, fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                  fwd/bwd
                </span>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
