import React, { useEffect, useMemo } from "react";
import { useLandscapeStore } from "../../store/landscapeStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { useDatasetStore } from "../../store/datasetStore";
import { GRADIENT_RAMP_DARK, renderHeatmap } from "../../utils/colorRamps";
import { ThreeScene } from "./three_d/ThreeScene";
import { Surface3DRenderer } from "./three_d/Surface3DRenderer";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function LandscapeView() {
  const graphId = useSimulatorStore((s) => s.graphId);
  const datasetId = useDatasetStore((s) => s.datasetId);
  const { status, progress, resolution, range, result, setResolution, setRange, compute, poll } = useLandscapeStore();
  const [viewMode, setViewMode] = React.useState<"heatmap" | "3d">("heatmap");

  useEffect(() => {
    if (status !== "computing") return;
    const handle = window.setInterval(() => {
      void poll();
    }, 500);
    return () => window.clearInterval(handle);
  }, [status, poll]);

  const heatmap = useMemo(() => {
    if (!result?.loss_surface?.length) return "";
    const height = result.loss_surface.length;
    const width = result.loss_surface[0]?.length ?? 0;
    if (!width || !height) return "";
    const flat = new Float32Array(width * height);
    let idx = 0;
    for (let i = 0; i < height; i += 1) {
      for (let j = 0; j < width; j += 1) {
        flat[idx] = result.loss_surface[i][j];
        idx += 1;
      }
    }
    return renderHeatmap(flat, width, height, GRADIENT_RAMP_DARK, true);
  }, [result]);

  const disabled = !graphId || !datasetId || status === "computing";

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Loss Landscape</div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Resolution</label>
            <NeuralInput type="number" min={10} max={100} value={resolution} onChange={(e) => setResolution(Number(e.target.value))} />
          </div>
          <div className="view-field">
            <label className="view-label">Range</label>
            <NeuralInput type="number" step={0.5} min={0.5} max={5} value={range} onChange={(e) => setRange(Number(e.target.value))} />
          </div>
          <NeuralButton
            variant="primary"
            disabled={disabled}
            onClick={() => {
              if (graphId && datasetId) void compute(graphId, datasetId);
            }}
          >
            {status === "computing" ? "Computing..." : "Compute"}
          </NeuralButton>
          <NeuralButton variant="secondary" onClick={() => setViewMode(viewMode === "3d" ? "heatmap" : "3d")}>
            {viewMode === "3d" ? "Heatmap" : "3D"}
          </NeuralButton>
        </div>
        {status === "computing" && <div className="view-status">Progress: {(progress * 100).toFixed(1)}%</div>}
        {(!graphId || !datasetId) && <div className="view-warning">Build a graph and load a dataset to compute the landscape.</div>}
      </NeuralPanel>

      <div className="view-grid">
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Surface Preview</div>
          {viewMode === "3d" && result ? (
            <ThreeScene height={360}>
              <Surface3DRenderer gridX={result.grid_x} gridY={result.grid_y} lossSurface={result.loss_surface} heightScale={1.2} />
            </ThreeScene>
          ) : heatmap ? (
            <div className="view-image-wrap">
              <img src={heatmap} alt="Loss landscape heatmap" className="view-image" />
            </div>
          ) : (
            <div className="view-empty">No surface computed yet.</div>
          )}
        </NeuralPanel>
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Stats</div>
          {result ? (
            <div className="view-stack">
              <div className="view-text">Center loss: {result.center_loss.toFixed(4)}</div>
              <div className="view-text">Min loss: {result.min_loss.toFixed(4)}</div>
              <div className="view-text">Max loss: {result.max_loss.toFixed(4)}</div>
              <div className="view-text">Sharpness: {result.sharpness_score.toFixed(4)}</div>
              <div className="view-text">Flatness: {result.flatness_description}</div>
            </div>
          ) : (
            <div className="view-empty">No stats available.</div>
          )}
        </NeuralPanel>
      </div>
    </div>
  );
}
