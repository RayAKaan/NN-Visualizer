import React, { useMemo } from "react";
import { useTrainingSimStore } from "../../store/trainingSimStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { GlowLineChart } from "@/design-system/charts/GlowLineChart";
import { neuralPalette } from "@/design-system/tokens/colors";

export function LiveMetricsView() {
  const metrics = useTrainingSimStore((s) => s.metricsHistory);
  const latest = metrics[metrics.length - 1];

  const trainLoss = useMemo(() => metrics.map((m) => m.train_loss), [metrics]);
  const testLoss = useMemo(() => metrics.map((m) => m.test_loss), [metrics]);
  const trainAcc = useMemo(() => metrics.map((m) => m.train_accuracy * 100), [metrics]);
  const testAcc = useMemo(() => metrics.map((m) => m.test_accuracy * 100), [metrics]);

  return (
    <div className="metrics-view">
      <NeuralPanel className="metrics-summary" variant="base">
        <div className="metrics-title">Live Metrics</div>
        {latest ? (
          <div className="metrics-grid">
            <div>
              <span>Train Loss</span>
              <strong>{latest.train_loss.toFixed(4)}</strong>
            </div>
            <div>
              <span>Test Loss</span>
              <strong>{latest.test_loss.toFixed(4)}</strong>
            </div>
            <div>
              <span>Train Acc</span>
              <strong>{(latest.train_accuracy * 100).toFixed(1)}%</strong>
            </div>
            <div>
              <span>Test Acc</span>
              <strong>{(latest.test_accuracy * 100).toFixed(1)}%</strong>
            </div>
            <div>
              <span>LR</span>
              <strong>{latest.learning_rate.toFixed(4)}</strong>
            </div>
            <div>
              <span>Epoch</span>
              <strong>{metrics.length}</strong>
            </div>
          </div>
        ) : (
          <div className="metrics-empty">Start training to see metrics.</div>
        )}
      </NeuralPanel>

      <div className="metrics-charts">
        <NeuralPanel className="metrics-chart-panel" variant="base">
          <GlowLineChart values={trainLoss} color={neuralPalette.lesion.bright} label="Train Loss" />
        </NeuralPanel>
        <NeuralPanel className="metrics-chart-panel" variant="base">
          <GlowLineChart values={testLoss} color={neuralPalette.cortex.bright} label="Test Loss" />
        </NeuralPanel>
        <NeuralPanel className="metrics-chart-panel" variant="base">
          <GlowLineChart values={trainAcc} color={neuralPalette.soma.bright} label="Train Accuracy (%)" />
        </NeuralPanel>
        <NeuralPanel className="metrics-chart-panel" variant="base">
          <GlowLineChart values={testAcc} color={neuralPalette.axon.bright} label="Test Accuracy (%)" />
        </NeuralPanel>
      </div>
    </div>
  );
}
