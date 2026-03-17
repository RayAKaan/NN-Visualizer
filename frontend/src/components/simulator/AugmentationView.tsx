import React from "react";
import { useAugmentationStore } from "../../store/augmentationStore";
import { useDatasetStore } from "../../store/datasetStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

function inferSquareShape(length: number): number[] | null {
  const side = Math.sqrt(length);
  if (Number.isInteger(side)) return [1, side, side];
  return null;
}

export function AugmentationView() {
  const trainData = useDatasetStore((s) => s.trainData);
  const { loading, nSamples, pipeline, result, setNSamples, setPipeline, preview } = useAugmentationStore();

  const sample = trainData[0];
  const input = sample?.x ?? null;
  const shape = input ? inferSquareShape(input.length) : null;

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Augmentation Playground</div>
        <div className="view-actions">
          <div className="view-field view-checkboxes">
            <label className="view-checkbox">
              <input
                type="checkbox"
                checked={pipeline.some((p) => p.type === "noise")}
                onChange={(e) => {
                  const next = pipeline.filter((p) => p.type !== "noise");
                  if (e.target.checked) next.push({ type: "noise", sigma: 0.1 });
                  setPipeline(next);
                }}
              />
              Noise
            </label>
            <label className="view-checkbox">
              <input
                type="checkbox"
                checked={pipeline.some((p) => p.type === "flip_h")}
                onChange={(e) => {
                  const next = pipeline.filter((p) => p.type !== "flip_h");
                  if (e.target.checked) next.push({ type: "flip_h" });
                  setPipeline(next);
                }}
              />
              Flip
            </label>
            <label className="view-checkbox">
              <input
                type="checkbox"
                checked={pipeline.some((p) => p.type === "rotate90")}
                onChange={(e) => {
                  const next = pipeline.filter((p) => p.type !== "rotate90");
                  if (e.target.checked) next.push({ type: "rotate90", k: 1 });
                  setPipeline(next);
                }}
              />
              Rotate90
            </label>
            <label className="view-checkbox">
              <input
                type="checkbox"
                checked={pipeline.some((p) => p.type === "crop")}
                onChange={(e) => {
                  const next = pipeline.filter((p) => p.type !== "crop");
                  if (e.target.checked) next.push({ type: "crop", ratio: 0.8 });
                  setPipeline(next);
                }}
              />
              Crop
            </label>
          </div>
          <div className="view-field">
            <label className="view-label">Samples</label>
            <NeuralInput type="number" min={1} max={16} value={nSamples} onChange={(e) => setNSamples(Number(e.target.value))} />
          </div>
          <NeuralButton variant="primary" disabled={!input || loading} onClick={() => input && preview(input, shape)}>
            {loading ? "Previewing..." : "Preview"}
          </NeuralButton>
        </div>
        {!input && <div className="view-warning">Load a dataset sample to preview augmentations.</div>}
      </NeuralPanel>

      <NeuralPanel className="view-panel" variant="base">
        <div className="view-subtitle">Previews</div>
        <div className="view-grid">
          {(result?.samples ?? []).map((img, idx) => (
            <img key={idx} src={img} alt={`aug-${idx}`} className="view-image" />
          ))}
          {!result && <div className="view-empty">No previews yet.</div>}
        </div>
      </NeuralPanel>
    </div>
  );
}
