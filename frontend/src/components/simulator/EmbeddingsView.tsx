import React, { useMemo } from "react";
import { useEmbeddingStore } from "../../store/embeddingStore";
import { useDatasetStore } from "../../store/datasetStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { ThreeScene } from "./three_d/ThreeScene";
import { ScatterPlot3DRenderer } from "./three_d/ScatterPlot3DRenderer";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { neuralPalette } from "@/design-system/tokens/colors";

const COLORS = [
  neuralPalette.axon.bright,
  neuralPalette.myelin.bright,
  neuralPalette.soma.bright,
  neuralPalette.cortex.bright,
  neuralPalette.dendrite.bright,
  neuralPalette.lesion.bright,
  neuralPalette.synapse.bright,
  neuralPalette.cloud,
];

export function EmbeddingsView() {
  const graphId = useSimulatorStore((s) => s.graphId);
  const datasetId = useDatasetStore((s) => s.datasetId);
  const { result, loading, layerIndex, nSamples, setLayerIndex, setNSamples, compute } = useEmbeddingStore();

  const [viewMode, setViewMode] = React.useState<"2d" | "3d">("2d");
  const points = result?.projections ?? [];
  const bounds = useMemo(() => {
    if (points.length === 0) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of points) {
      const [x, y] = p.coords;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return { minX, maxX, minY, maxY };
  }, [points]);

  const disabled = !graphId || !datasetId || loading;

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Embedding Explorer</div>
        <div className="view-actions">
          <NeuralButton variant={viewMode === "2d" ? "primary" : "secondary"} onClick={() => setViewMode("2d")}>
            2D
          </NeuralButton>
          <NeuralButton variant={viewMode === "3d" ? "primary" : "secondary"} onClick={() => setViewMode("3d")}>
            3D
          </NeuralButton>
          <div className="view-field">
            <label className="view-label">Layer</label>
            <NeuralInput type="number" min={0} value={layerIndex} onChange={(e) => setLayerIndex(Number(e.target.value))} />
          </div>
          <div className="view-field">
            <label className="view-label">Samples</label>
            <NeuralInput type="number" min={20} max={500} value={nSamples} onChange={(e) => setNSamples(Number(e.target.value))} />
          </div>
          <NeuralButton
            variant="primary"
            disabled={disabled}
            onClick={() => {
              if (graphId && datasetId) void compute(graphId, datasetId);
            }}
          >
            {loading ? "Computing..." : "Compute"}
          </NeuralButton>
        </div>
        {(!graphId || !datasetId) && <div className="view-warning">Build a graph and load a dataset to compute embeddings.</div>}
      </NeuralPanel>

      <NeuralPanel className="view-panel" variant="base">
        <div className="view-subtitle">Projection</div>
        {viewMode === "3d" ? (
          <ThreeScene height={320}>
            <ScatterPlot3DRenderer points={points} />
          </ThreeScene>
        ) : (
          <svg width="100%" height="320" viewBox="0 0 600 320" className="view-svg">
            {points.map((p) => {
              const [x, y] = p.coords;
              const nx = (x - bounds.minX) / (bounds.maxX - bounds.minX || 1);
              const ny = (y - bounds.minY) / (bounds.maxY - bounds.minY || 1);
              const cx = 40 + nx * 520;
              const cy = 280 - ny * 240;
              const color = COLORS[p.label % COLORS.length];
              return <circle key={p.index} cx={cx} cy={cy} r={3.5} fill={color} opacity={0.85} />;
            })}
          </svg>
        )}
      </NeuralPanel>

      <NeuralPanel className="view-panel" variant="base">
        <div className="view-subtitle">Stats</div>
        {result ? (
          <div className="view-stack">
            <div className="view-text">Variance explained: {result.variance_explained.map((v) => v.toFixed(3)).join(", ")}</div>
            <div className="view-text">Silhouette score: {result.cluster_metrics.silhouette_score.toFixed(3)}</div>
            <div className="view-text">Inter-class distance: {result.cluster_metrics.inter_class_distance.toFixed(3)}</div>
            <div className="view-text">Intra-class distance: {result.cluster_metrics.intra_class_distance.toFixed(3)}</div>
          </div>
        ) : (
          <div className="view-empty">No stats available.</div>
        )}
      </NeuralPanel>
    </div>
  );
}
