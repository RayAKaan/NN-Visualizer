import React, { useEffect, useMemo, useRef, useState } from "react";
import { useComputationStore } from "../../store/computationStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { simulatorApi } from "../../hooks/useSimulatorApi";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { weightColor, gradientHealthColor, neuralPalette } from "@/design-system/tokens/colors";

function drawHeatmap(canvas: HTMLCanvasElement, matrix: number[][]) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!rows || !cols) return;
  const cellW = rect.width / cols;
  const cellH = rect.height / rows;
  let maxAbs = 0;
  matrix.forEach((r) => r.forEach((v) => (maxAbs = Math.max(maxAbs, Math.abs(v)))));
  maxAbs = maxAbs || 1;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const v = matrix[r][c];
      const color = weightColor(v, maxAbs);
      ctx.fillStyle = color;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
      ctx.strokeStyle = "rgba(36,40,54,0.5)";
      ctx.lineWidth = 0.5;
      ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);
      if (Math.abs(v) > maxAbs * 0.9) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        ctx.shadowBlur = 0;
      }
    }
  }
}

function drawHistogram(canvas: HTMLCanvasElement, bins: number[], counts: number[], color: string) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const maxCount = Math.max(...counts, 1);
  const barW = rect.width / Math.max(counts.length, 1);
  counts.forEach((count, i) => {
    const h = (count / maxCount) * (rect.height - 16);
    ctx.fillStyle = `${color}B3`;
    ctx.fillRect(i * barW, rect.height - h, barW - 1, h);
  });
}

export function InspectorView() {
  const weightInspection = useComputationStore((s) => s.weightInspection);
  const activationInspection = useComputationStore((s) => s.activationInspection);
  const graphId = useSimulatorStore((s) => s.graphId);
  const [flow, setFlow] = useState<any>(null);
  const heatmapRef = useRef<HTMLCanvasElement>(null);
  const histRef = useRef<HTMLCanvasElement>(null);
  const actHistRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!graphId) return;
    void (async () => {
      try {
        const res = await simulatorApi.gradientFlow(graphId);
        setFlow(res);
      } catch {
        setFlow(null);
      }
    })();
  }, [graphId]);

  useEffect(() => {
    if (heatmapRef.current && weightInspection?.weight_matrix) {
      drawHeatmap(heatmapRef.current, weightInspection.weight_matrix);
    }
  }, [weightInspection]);

  useEffect(() => {
    if (histRef.current && weightInspection?.histogram) {
      drawHistogram(histRef.current, weightInspection.histogram.bins, weightInspection.histogram.counts, neuralPalette.synapse.bright);
    }
  }, [weightInspection]);

  useEffect(() => {
    if (actHistRef.current && activationInspection?.histogram) {
      drawHistogram(actHistRef.current, activationInspection.histogram.bins, activationInspection.histogram.counts, neuralPalette.axon.bright);
    }
  }, [activationInspection]);

  const gradNorm = useMemo(() => {
    const norms = flow?.per_layer?.map((p: any) => p.gradient_norm) ?? [];
    return norms.reduce((acc: number, v: number) => acc + v, 0);
  }, [flow]);

  return (
    <div className="inspector-view">
      <NeuralPanel className="inspector-header" variant="base">
        <div className="inspector-title">Inspector</div>
        <div className="inspector-subtitle">Weights, activations, gradients</div>
      </NeuralPanel>

      <div className="inspector-grid">
        <NeuralPanel className="inspector-panel" variant="base">
          <div className="inspector-panel-title">Weight Heatmap</div>
          {weightInspection ? (
            <>
              <canvas ref={heatmapRef} className="inspector-heatmap" />
              <div className="inspector-stats">
                <span>Mean: {weightInspection.stats.mean.toFixed(3)}</span>
                <span>Std: {weightInspection.stats.std.toFixed(3)}</span>
                <span>Min: {weightInspection.stats.min.toFixed(3)}</span>
                <span>Max: {weightInspection.stats.max.toFixed(3)}</span>
              </div>
            </>
          ) : (
            <div className="inspector-empty">No weight data.</div>
          )}
        </NeuralPanel>

        <NeuralPanel className="inspector-panel" variant="base">
          <div className="inspector-panel-title">Weight Histogram</div>
          {weightInspection ? (
            <>
              <canvas ref={histRef} className="inspector-histogram" />
              <div className="inspector-stats">
                <span>||W||: {weightInspection.stats.l2_norm.toFixed(3)}</span>
                <span>Sparsity: {(weightInspection.stats.sparsity * 100).toFixed(1)}%</span>
              </div>
            </>
          ) : (
            <div className="inspector-empty">No histogram data.</div>
          )}
        </NeuralPanel>

        <NeuralPanel className="inspector-panel" variant="base">
          <div className="inspector-panel-title">Activations</div>
          {activationInspection ? (
            <>
              <div className="inspector-stats">
                <span>Activation: {activationInspection.activation_name}</span>
                <span>Dead neurons: {activationInspection.dead_neurons.length}</span>
              </div>
              <canvas ref={actHistRef} className="inspector-histogram" />
            </>
          ) : (
            <div className="inspector-empty">No activation data.</div>
          )}
        </NeuralPanel>

        <NeuralPanel className="inspector-panel" variant="base">
          <div className="inspector-panel-title">Gradient Flow</div>
          {flow ? (
            <>
              <div className="inspector-stats">
                <span>Total ||g||: {Number(flow.total_gradient_norm ?? gradNorm).toFixed(3)}</span>
                <span>Flow ratio: {(Number(flow.flow_ratio ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="inspector-gradient">
              {(flow.per_layer ?? []).map((p: any, idx: number) => (
                <div key={idx} className="inspector-gradient-row">
                  <span>Layer {idx + 1}</span>
                  <div className="inspector-gradient-bar">
                    <div
                      className="inspector-gradient-fill"
                      style={{
                          width: `${Math.min(100, (p.gradient_norm ?? 0) * 10)}%`,
                          background: gradientHealthColor(p.gradient_norm ?? 0),
                        }}
                      />
                    </div>
                    <span>{Number(p.gradient_norm ?? 0).toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="inspector-empty">No gradient flow yet.</div>
          )}
        </NeuralPanel>
      </div>
    </div>
  );
}
