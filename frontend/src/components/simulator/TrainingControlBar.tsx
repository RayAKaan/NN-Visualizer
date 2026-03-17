import React, { useEffect, useMemo, useRef } from "react";
import { useSimulatorStore } from "../../store/simulatorStore";
import { useDatasetStore } from "../../store/datasetStore";
import { useTrainingSimStore } from "../../store/trainingSimStore";
import { useTrainingSimSocket } from "../../hooks/useTrainingSimSocket";
import { useBackpropStore } from "../../store/backpropStore";
import { useReplayStore } from "../../store/replayStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { NeuralSlider } from "@/design-system/components/NeuralSlider";
import { NeuralTabs } from "@/design-system/components/NeuralTabs";
import { NeuralNumber } from "@/design-system/components/NeuralNumber";
import { gradientHealthColor, neuralPalette } from "@/design-system/tokens/colors";

export function TrainingControlBar() {
  const graphId = useSimulatorStore((s) => s.graphId);
  const datasetId = useDatasetStore((s) => s.datasetId);
  const config = useTrainingSimStore((s) => s.config);
  const isTraining = useTrainingSimStore((s) => s.isTraining);
  const isPaused = useTrainingSimStore((s) => s.isPaused);
  const metricsHistory = useTrainingSimStore((s) => s.metricsHistory);
  const currentEpoch = useTrainingSimStore((s) => s.currentEpoch);
  const totalEpochs = useTrainingSimStore((s) => s.totalEpochs);
  const { isConnected, send } = useTrainingSimSocket();
  const lastConfig = useRef(config);
  const mode = useBackpropStore((s) => s.mode);
  const switchMode = useBackpropStore((s) => s.switchMode);
  const playbackSpeed = useReplayStore((s) => s.playbackSpeed);
  const setPlaybackSpeed = useReplayStore((s) => s.setSpeed);
  const snapshots = useReplayStore((s) => s.snapshots);
  const currentSnapshotIndex = useReplayStore((s) => s.currentSnapshotIndex);
  const loadSnapshot = useReplayStore((s) => s.loadSnapshot);
  const animationSpeed = useSimulatorStore((s) => s.animationSpeed);
  const setAnimationSpeed = useSimulatorStore((s) => s.setAnimationSpeed);

  const start = () => {
    if (!graphId || !datasetId) return;
    send({ action: "start", graph_id: graphId, dataset_id: datasetId, config });
  };
  const pause = () => send({ action: "pause" });
  const resume = () => send({ action: "resume" });
  const stop = () => send({ action: "stop" });

  useEffect(() => {
    if (!isTraining || !graphId) {
      lastConfig.current = config;
      return;
    }
    if (lastConfig.current !== config) {
      send({ action: "update_config", graph_id: graphId, updates: config });
      lastConfig.current = config;
    }
  }, [config, graphId, isTraining, send]);

  const latest = metricsHistory[metricsHistory.length - 1];
  const previous = metricsHistory[metricsHistory.length - 2];
  const lossTrend = latest && previous ? latest.train_loss - previous.train_loss : 0;
  const lossColor = lossTrend < 0 ? neuralPalette.soma.bright : lossTrend > 0 ? neuralPalette.lesion.bright : neuralPalette.cortex.bright;
  const gradNorm = latest?.gradient_norms?.reduce((acc, v) => acc + v, 0) ?? 0;
  const gradientColor = gradientHealthColor(gradNorm || 0);
  const modeTabs = useMemo(
    () => [
      { id: "forward", label: "Fwd" },
      { id: "backward", label: "Bwd" },
      { id: "full_cycle", label: "Train" },
    ],
    [],
  );

  return (
    <NeuralPanel className="train-bar" variant="base">
      <div className="train-bar-inner">
        <div className="train-bar-left">
          <div className="train-connection">
            <span className={`train-dot ${isConnected ? "on" : "off"}`} />
            <span>Training WS: {isConnected ? "connected" : "offline"}</span>
          </div>
          {isTraining && (
            <div className="train-active">
              <span className="train-active-dot" style={{ background: lossColor }} />
              <span>Training</span>
            </div>
          )}
          <NeuralTabs
            tabs={modeTabs}
            value={mode}
            onChange={(id) => switchMode(id as "forward" | "backward" | "full_cycle")}
            className="train-mode-tabs"
          />
        </div>

        <div className="train-bar-controls">
          {!isTraining && (
            <NeuralButton variant="primary" onClick={start}>
              Start
            </NeuralButton>
          )}
          {isTraining && !isPaused && (
            <NeuralButton variant="secondary" onClick={pause}>
              Pause
            </NeuralButton>
          )}
          {isTraining && isPaused && (
            <NeuralButton variant="primary" onClick={resume}>
              Resume
            </NeuralButton>
          )}
          {isTraining && (
            <NeuralButton variant="danger" onClick={stop}>
              Stop
            </NeuralButton>
          )}
        </div>

        <div className="train-sliders">
          <label className="train-slider">
            <span>Speed</span>
            <NeuralSlider
              min={0.5}
              max={3}
              step={0.1}
              value={animationSpeed}
              onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
            />
            <NeuralNumber value={animationSpeed} precision={1} />
          </label>
          <label className="train-slider">
            <span>Replay</span>
            <NeuralSlider
              min={0}
              max={Math.max(0, snapshots.length - 1)}
              step={1}
              value={currentSnapshotIndex}
              onChange={(e) => void loadSnapshot(parseInt(e.target.value))}
              disabled={snapshots.length === 0}
            />
            <span className="train-slider-value">
              {snapshots.length ? `E${snapshots[currentSnapshotIndex]?.epoch ?? currentSnapshotIndex}` : "n/a"}
            </span>
          </label>
          <label className="train-slider">
            <span>Playback</span>
            <NeuralSlider min={0.5} max={6} step={0.5} value={playbackSpeed} onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))} />
            <NeuralNumber value={playbackSpeed} precision={1} />
          </label>
        </div>

        <div className="train-metrics">
          <div className="train-metric">
            <span>Epoch</span>
            <span className="train-metric-value">
              {currentEpoch}
              {totalEpochs ? ` / ${totalEpochs}` : ""}
            </span>
          </div>
          <div className="train-metric">
            <span>Loss</span>
            <span className="train-metric-value" style={{ color: lossColor }}>
              {latest ? <NeuralNumber value={latest.train_loss} precision={4} /> : "n/a"}
            </span>
          </div>
          <div className="train-metric">
            <span>Accuracy</span>
            <span className="train-metric-value" style={{ color: neuralPalette.soma.bright }}>
              {latest ? <NeuralNumber value={latest.train_accuracy * 100} precision={1} /> : "n/a"}%
            </span>
          </div>
          <div className="train-metric">
            <span>||g||</span>
            <span className="train-metric-value" style={{ color: gradientColor }}>
              {latest ? <NeuralNumber value={gradNorm} precision={3} /> : "n/a"}
            </span>
          </div>
        </div>
      </div>
    </NeuralPanel>
  );
}
